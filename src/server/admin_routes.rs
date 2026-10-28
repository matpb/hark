//! `/hark/v1/models`: catalog, pull, load, unload and delete.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use axum::Json;
use axum::body::Bytes;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use serde_json::{Value, json};

use super::{ApiError, AppState, engine_error, internal};
use crate::engine::ModelRef;
use crate::models::{CATALOG, DeleteOutcome, ModelError, ModelKind};
use crate::pull::{self, HfFetch, PullError, PullOutcome, PullSpec};

pub(super) const MAX_BODY: usize = 16 * 1024;

/// Admin operations per model id: pulls share the lock, delete, load and unload hold it alone.
#[derive(Default)]
pub(super) struct ModelLocks {
    map: Mutex<HashMap<String, Arc<tokio::sync::RwLock<()>>>>,
}

impl ModelLocks {
    fn entry(&self, id: &str) -> Arc<tokio::sync::RwLock<()>> {
        self.map
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(id.to_string())
            .or_default()
            .clone()
    }

    pub(super) async fn shared(&self, id: &str) -> tokio::sync::OwnedRwLockReadGuard<()> {
        self.entry(id).read_owned().await
    }

    pub(super) async fn exclusive(&self, id: &str) -> tokio::sync::OwnedRwLockWriteGuard<()> {
        self.entry(id).write_owned().await
    }
}

/// Marks a model as being deleted for as long as it lives, so no request loads it again meanwhile.
struct Deleting {
    engine: Arc<dyn crate::engine::Transcriber>,
    id: &'static str,
}

impl Deleting {
    fn start(engine: Arc<dyn crate::engine::Transcriber>, id: &'static str) -> Deleting {
        engine.set_deleting(id, true);
        Deleting { engine, id }
    }
}

impl Drop for Deleting {
    fn drop(&mut self) {
        self.engine.set_deleting(self.id, false);
    }
}

fn model_error(e: ModelError) -> ApiError {
    ApiError::model(e, "id")
}

fn pull_error(e: PullError) -> ApiError {
    let (status, kind, code) = match &e {
        PullError::Io(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            "server_error",
            "model_store_error",
        ),
        PullError::Timeout(_) => (
            StatusCode::GATEWAY_TIMEOUT,
            "server_error",
            "model_download_timeout",
        ),
        PullError::Network(_) | PullError::Mismatch(_) => (
            StatusCode::BAD_GATEWAY,
            "server_error",
            "model_download_failed",
        ),
    };
    ApiError::new(status, kind, code, e.to_string())
}

/// Every catalog entry with whether it is installed and loaded.
pub(super) async fn catalog(State(state): State<AppState>) -> Json<Value> {
    let status = state.engine.status();
    let data: Vec<Value> = CATALOG
        .iter()
        .map(|e| {
            let located = state.registry.locate(e);
            let own = state.registry.models_dir().join(e.file);
            let loaded = status.models.iter().find(|m| m.id == e.id);
            json!({
                "id": e.id,
                "file": e.file,
                "size_bytes": e.size,
                "sha256": e.sha256,
                "kind": match e.kind { ModelKind::Transcription => "transcription", ModelKind::Vad => "vad" },
                "installed": located.is_some(),
                "location": match &located {
                    None => Value::Null,
                    Some(p) if *p == own => json!("models_dir"),
                    Some(_) => json!("extra_dir"),
                },
                "loaded": loaded.is_some(),
                "active": loaded.is_some_and(|m| m.active),
            })
        })
        .collect();
    Json(json!({"object": "list", "data": data}))
}

pub(super) async fn pull(
    State(state): State<AppState>,
    body: Bytes,
) -> Result<Json<Value>, ApiError> {
    let bad = |m: &str| ApiError::invalid("id", "invalid_value", m.to_string());
    let v: Value = serde_json::from_slice(&body)
        .map_err(|_| bad("send a JSON body like {\"id\": \"large-v3-turbo\"}"))?;
    let id = v["id"]
        .as_str()
        .filter(|s| s.len() <= 64)
        .ok_or_else(|| bad("`id` must be a model id string"))?
        .to_string();
    let entry = state.registry.resolve_pullable(&id).map_err(model_error)?;
    let _admin = state.admin.shared(entry.id).await;
    let registry = state.registry.clone();
    let (result, leader) = tokio::task::spawn_blocking(move || {
        if registry.locate(entry).is_some() {
            return (Ok(PullOutcome::AlreadyInstalled), false);
        }
        let dir = registry.models_dir().to_path_buf();
        let key = pull::pull_key(&dir, entry.file);
        let spec = PullSpec::from_entry(entry);
        pull::global().run(&key, || {
            tracing::info!(
                model = entry.id,
                size = entry.size,
                "model download started"
            );
            let r = pull::pull(&dir, &spec, &HfFetch::default());
            match &r {
                Ok(_) => tracing::info!(model = entry.id, "model downloaded"),
                Err(e) => tracing::warn!(model = entry.id, error = %e, "model download failed"),
            }
            r
        })
    })
    .await
    .map_err(internal)?;
    let outcome = result.map_err(pull_error)?;
    Ok(Json(json!({
        "id": entry.id,
        "file": entry.file,
        "size_bytes": entry.size,
        "status": match outcome {
            PullOutcome::AlreadyInstalled => "already_installed",
            PullOutcome::Downloaded { .. } => "installed",
        },
        "downloaded": matches!(outcome, PullOutcome::Downloaded { .. }),
        "shared_download": !leader && matches!(outcome, PullOutcome::Downloaded { .. }),
    })))
}

pub(super) async fn load(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = state.registry.resolve(&id).map_err(model_error)?;
    let _admin = state.admin.exclusive(entry.id).await;
    let (entry, path) = state.registry.resolve_installed(&id).map_err(model_error)?;
    let engine = state.engine.clone();
    let model = ModelRef { id: entry.id, path };
    tokio::task::spawn_blocking(move || engine.load_model(&model))
        .await
        .map_err(internal)?
        .map_err(|e| engine_error(e, &super::new_request_id()))?;
    Ok(Json(json!({"id": entry.id, "loaded": true})))
}

pub(super) async fn unload(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = state.registry.resolve(&id).map_err(model_error)?;
    let _admin = state.admin.exclusive(entry.id).await;
    let engine = state.engine.clone();
    let was = tokio::task::spawn_blocking(move || engine.unload_model(entry.id))
        .await
        .map_err(internal)?
        .map_err(|e| engine_error(e, &super::new_request_id()))?;
    Ok(Json(
        json!({"id": entry.id, "unloaded": was, "loaded": false}),
    ))
}

pub(super) async fn remove(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let entry = state.registry.lookup_catalog(&id).map_err(model_error)?;
    let _admin = state.admin.exclusive(entry.id).await;
    let own = state.registry.models_dir().join(entry.file);
    if !own.is_file() {
        return Err(match state.registry.locate(entry) {
            Some(_) => ApiError::new(
                StatusCode::CONFLICT,
                "invalid_request_error",
                "model_read_only",
                format!(
                    "`{}` is installed in a read-only extra directory (models.extra_dirs); hark only deletes from models.dir",
                    entry.id
                ),
            ),
            None => model_error(ModelError::NotInstalled(entry.id.to_string())),
        });
    }
    let engine = state.engine.clone();
    let registry = state.registry.clone();
    let outcome = tokio::task::spawn_blocking(move || {
        let _deleting = Deleting::start(engine.clone(), entry.id);
        if entry.kind == ModelKind::Transcription {
            engine
                .unload_model(entry.id)
                .map_err(|e| engine_error(e, &super::new_request_id()))?;
        }
        registry.delete_from_models_dir(entry).map_err(|e| {
            ApiError::new(
                StatusCode::INTERNAL_SERVER_ERROR,
                "server_error",
                "model_store_error",
                format!("could not delete {}: {e}", entry.file),
            )
        })
    })
    .await
    .map_err(internal)??;
    match outcome {
        DeleteOutcome::Removed => Ok(Json(json!({"id": entry.id, "deleted": true}))),
        _ => Err(model_error(ModelError::NotInstalled(entry.id.to_string()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pull_past_its_deadline_is_a_504_envelope() {
        let e = pull_error(PullError::Timeout("ggml-x.bin: too slow".into()));
        assert_eq!(e.status, StatusCode::GATEWAY_TIMEOUT);
        assert_eq!(e.code, "model_download_timeout");
        assert_eq!(e.kind, "server_error");
        let e = pull_error(PullError::Network("x".into()));
        assert_eq!(e.status, StatusCode::BAD_GATEWAY);
    }
}
