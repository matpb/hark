use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use clap::{Parser, Subcommand};
use hark::cli::{Client, SubmitArgs, TranscribeArgs};
use hark::config::{Config, default_config_path};
use hark::engine::{ModelRef, Transcriber, WhisperEngine};
use hark::jobs::JobManager;
use hark::models::Registry;
use hark::server::{self, AppState};
use hark::vocabulary::VocabularyStore;
use tracing_subscriber::EnvFilter;

#[derive(Parser)]
#[command(
    name = "hark",
    version,
    about = "Local OpenAI-compatible transcription daemon"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Run the daemon.
    Serve {
        /// Config file; a missing file means all defaults.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Append logs to this file; overrides `server.log_file`.
        #[arg(long)]
        log_file: Option<PathBuf>,
    },
    /// Transcribe a file through the running daemon, or in this process with --local.
    Transcribe {
        file: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        language: Option<String>,
        /// json, text (default), srt, vtt or verbose_json.
        #[arg(long)]
        format: Option<String>,
        #[arg(long)]
        vocabulary: Option<String>,
        /// interactive (default) or batch.
        #[arg(long)]
        priority: Option<String>,
        /// Run in this process without the daemon.
        #[arg(long)]
        local: bool,
        /// Config file; the daemon address and key file come from it.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Manage models on the running daemon.
    Models {
        /// Config file; the daemon address and key file come from it.
        #[arg(long, global = true)]
        config: Option<PathBuf>,
        #[command(subcommand)]
        command: ModelsCommand,
    },
    /// Show the running daemon's status.
    Status {
        /// Config file; the daemon address and key file come from it.
        #[arg(long)]
        config: Option<PathBuf>,
    },
    /// Check ffmpeg, Vulkan devices, models, config, port and the systemd unit.
    Doctor {
        /// Config file to check.
        #[arg(long)]
        config: Option<PathBuf>,
        /// Hash every installed model, not only the default and VAD models.
        #[arg(long)]
        all: bool,
    },
    /// Manage jobs on the running daemon.
    Jobs {
        /// Config file; the daemon address and path roots come from it.
        #[arg(long, global = true)]
        config: Option<PathBuf>,
        #[command(subcommand)]
        command: JobsCommand,
    },
}

#[derive(Subcommand)]
enum ModelsCommand {
    /// The catalog with what is installed and loaded.
    List,
    /// Download a catalog model into models.dir.
    Pull {
        /// Catalog id or alias.
        #[arg(required_unless_present = "defaults", conflicts_with = "defaults")]
        id: Option<String>,
        /// Pull the default model and the VAD model from the config.
        #[arg(long)]
        defaults: bool,
    },
    /// Delete a model from models.dir.
    Rm { id: String },
    /// Load a model now.
    Load { id: String },
    /// Unload a model once its states are idle.
    Unload { id: String },
}

#[derive(Subcommand)]
enum JobsCommand {
    /// Submit an audio file as a job.
    Submit {
        file: PathBuf,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        language: Option<String>,
        /// Default response format for the job's result.
        #[arg(long)]
        format: Option<String>,
        #[arg(long)]
        vocabulary: Option<String>,
    },
    /// List jobs, newest first.
    List {
        #[arg(long)]
        status: Option<String>,
    },
    /// Show one job.
    Get { id: String },
    /// Print a finished job's transcript.
    Result {
        id: String,
        #[arg(long)]
        format: Option<String>,
    },
    /// Cancel a running or queued job, or delete a finished one.
    Cancel { id: String },
}

fn main() -> anyhow::Result<()> {
    // Per-thread glibc arenas fragment under mixed load and grew RSS by 100 MiB in 15 min; two arenas stay flat.
    hark::sys::limit_malloc_arenas();
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

/// A shared append-mode log file; each event is one write.
struct SharedFile(Arc<std::fs::File>);

impl std::io::Write for SharedFile {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        (&*self.0).write(buf)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        (&*self.0).flush()
    }
}

/// The `--log-file` flag, else `server.log_file` from the config (read early; load errors surface later in `serve`).
fn log_file_path(
    config: Option<&std::path::Path>,
    flag: Option<&std::path::Path>,
) -> Option<PathBuf> {
    if let Some(p) = flag {
        return Some(p.to_path_buf());
    }
    let path = config
        .map(PathBuf::from)
        .unwrap_or_else(default_config_path);
    let cfg = Config::load(&path).ok()?;
    let f = cfg.server.log_file.trim();
    (!f.is_empty()).then(|| hark::config::expand_tilde(f))
}

fn open_log_file(path: PathBuf) -> Option<Arc<std::fs::File>> {
    use std::fs::OpenOptions;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut opts = OpenOptions::new();
    opts.create(true).append(true);
    #[cfg(unix)]
    std::os::unix::fs::OpenOptionsExt::mode(&mut opts, 0o600);
    match opts.open(&path) {
        Ok(f) => Some(Arc::new(f)),
        Err(e) => {
            eprintln!(
                "hark: cannot open log file {}: {e}; logging to stderr",
                path.display()
            );
            None
        }
    }
}

async fn run() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let default_level = if matches!(cli.command, Command::Doctor { .. }) {
        "warn"
    } else {
        "info"
    };
    use tracing_subscriber::Layer;
    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;
    let log_file = match &cli.command {
        Command::Serve { config, log_file } => {
            log_file_path(config.as_deref(), log_file.as_deref())
        }
        _ => None,
    }
    .and_then(open_log_file);
    let to_file = log_file.is_some();
    tracing_subscriber::registry()
        .with(
            {
                let l = tracing_subscriber::fmt::layer();
                if to_file { l.with_ansi(false) } else { l }
            }
            .with_writer(move || -> Box<dyn std::io::Write> {
                match &log_file {
                    Some(f) => Box::new(SharedFile(f.clone())),
                    None => Box::new(std::io::stderr()),
                }
            })
            .with_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(default_level)),
            ),
        )
        .with(
            hark::engine::BackendLogLayer.with_filter(tracing_subscriber::filter::filter_fn(
                hark::engine::backend_log_filter,
            )),
        )
        .init();
    match cli.command {
        Command::Doctor { config, all } => {
            let code = tokio::task::spawn_blocking(move || {
                whisper_rs::install_logging_hooks();
                hark::doctor::run_doctor(config, all)
            })
            .await?;
            // The GPU driver's teardown can hang at exit; the report is already printed.
            // SAFETY: _exit only terminates the process; nothing here needs a destructor.
            unsafe { libc::_exit(code) }
        }
        Command::Serve { config, .. } => serve(config).await,
        Command::Jobs { config, command } => {
            tokio::task::spawn_blocking(move || run_jobs(config, command)).await?
        }
        Command::Models { config, command } => {
            tokio::task::spawn_blocking(move || run_models(config, command)).await?
        }
        Command::Status { config } => {
            tokio::task::spawn_blocking(move || {
                let cfg = load_config(config)?;
                emit(Client::new(&cfg).status()?);
                Ok(())
            })
            .await?
        }
        Command::Transcribe {
            file,
            model,
            language,
            format,
            vocabulary,
            priority,
            local,
            config,
        } => {
            let args = TranscribeArgs {
                file,
                model,
                language,
                format,
                vocabulary,
                priority,
            };
            tokio::task::spawn_blocking(move || {
                let cfg = load_config(config)?;
                if local {
                    emit(hark::cli::transcribe_local(&cfg, &args)?);
                    // The GPU driver's teardown can hang at exit; the result is already printed.
                    // SAFETY: _exit only terminates the process; the output is already flushed.
                    unsafe { libc::_exit(0) }
                }
                emit(Client::new(&cfg).transcribe(&args)?);
                Ok(())
            })
            .await?
        }
    }
}

/// Startup order: signal handlers, bind, jobs store (takes its lock), preload, then READY and serving.
async fn serve(config: Option<PathBuf>) -> anyhow::Result<()> {
    let path = config.unwrap_or_else(default_config_path);
    let cfg = Config::load(&path).with_context(|| format!("loading {}", path.display()))?;
    let addr = cfg.listen_addr()?;
    hark::auth::load_key(&cfg.server.api_key_file).map_err(|e| anyhow::anyhow!(e))?;
    let shutdown = server::Shutdown::install().context("installing the signal handlers")?;
    let listener = server::bind(addr).await?;

    let registry = Arc::new(Registry::from_config(&cfg));
    let stale = hark::pull::remove_stale_parts(registry.models_dir());
    if stale > 0 {
        tracing::info!(count = stale, "removed stale partial model downloads");
    }
    let engine = Arc::new(WhisperEngine::from_config(&cfg, &registry));
    let cfg = Arc::new(cfg);
    let jobs = JobManager::open(cfg.clone(), registry.clone(), engine.clone())
        .context("opening the jobs directory")?;
    let vocabulary = Arc::new(VocabularyStore::new(&cfg.vocabulary));
    let _reloader = vocabulary
        .spawn_watcher(path, cfg.clone())
        .context("installing the SIGHUP handler")?;

    if cfg.engine.preload {
        match registry.resolve_installed(&cfg.engine.default_model) {
            Ok((entry, path)) => {
                let model = ModelRef { id: entry.id, path };
                let e = engine.clone();
                let loaded = tokio::task::spawn_blocking(move || e.preload(&model)).await?;
                if let Err(err) = loaded {
                    tracing::error!(error = %err, "preload failed; the model will load on first request");
                }
            }
            Err(err) => tracing::warn!(error = %err, "default model unavailable, skipping preload"),
        }
    }

    jobs.start();
    let state = AppState::new(cfg, registry, engine, vocabulary, jobs);
    server::serve_on(listener, state, shutdown).await?;
    // Job state is already fsynced; skipping exit handlers avoids a hang in the GPU driver's teardown.
    // SAFETY: _exit only terminates the process; job state is already on disk.
    unsafe { libc::_exit(0) }
}

fn load_config(config: Option<PathBuf>) -> anyhow::Result<Config> {
    let path = config.unwrap_or_else(default_config_path);
    Config::load(&path).with_context(|| format!("loading {}", path.display()))
}

fn emit(out: String) {
    print!("{out}");
    if !out.ends_with('\n') {
        println!();
    }
}

fn run_models(config: Option<PathBuf>, command: ModelsCommand) -> anyhow::Result<()> {
    let client = Client::new(&load_config(config)?);
    emit(match command {
        ModelsCommand::List => client.models_list()?,
        ModelsCommand::Pull { id: Some(id), .. } => client.models_pull(&id)?,
        ModelsCommand::Pull { .. } => client.models_pull_defaults()?,
        ModelsCommand::Rm { id } => client.models_rm(&id)?,
        ModelsCommand::Load { id } => client.models_load(&id)?,
        ModelsCommand::Unload { id } => client.models_unload(&id)?,
    });
    Ok(())
}

fn run_jobs(config: Option<PathBuf>, command: JobsCommand) -> anyhow::Result<()> {
    let path = config.unwrap_or_else(default_config_path);
    let cfg = Config::load(&path).with_context(|| format!("loading {}", path.display()))?;
    let client = Client::new(&cfg);
    let out = match command {
        JobsCommand::Submit {
            file,
            model,
            language,
            format,
            vocabulary,
        } => client.submit(&SubmitArgs {
            file,
            model,
            language,
            format,
            vocabulary,
        })?,
        JobsCommand::List { status } => client.list(status.as_deref())?,
        JobsCommand::Get { id } => client.get(&id)?,
        JobsCommand::Result { id, format } => client.result(&id, format.as_deref())?,
        JobsCommand::Cancel { id } => client.cancel(&id)?,
    };
    print!("{out}");
    if !out.ends_with('\n') {
        println!();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn models_pull_takes_an_id_or_defaults_but_not_both_or_neither() {
        assert!(Cli::try_parse_from(["hark", "models", "pull", "base"]).is_ok());
        assert!(Cli::try_parse_from(["hark", "models", "pull", "--defaults"]).is_ok());
        assert!(Cli::try_parse_from(["hark", "models", "pull"]).is_err());
        assert!(Cli::try_parse_from(["hark", "models", "pull", "base", "--defaults"]).is_err());
    }
}
