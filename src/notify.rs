//! systemd readiness notification (`sd_notify`) over `NOTIFY_SOCKET`; a no-op outside systemd.

#[cfg(target_os = "linux")]
use std::os::linux::net::SocketAddrExt;
#[cfg(target_os = "linux")]
use std::os::unix::net::{SocketAddr, UnixDatagram};

/// The datagram address for a `NOTIFY_SOCKET` value: a path, or `@name` for an abstract socket.
#[cfg(target_os = "linux")]
fn socket_addr(value: &str) -> std::io::Result<SocketAddr> {
    match value.strip_prefix('@') {
        Some(name) => SocketAddr::from_abstract_name(name.as_bytes()),
        None => SocketAddr::from_pathname(value),
    }
}

/// Sends `state` to the socket named by `value`; empty values are ignored.
#[cfg(target_os = "linux")]
pub fn send_to(value: &str, state: &str) -> std::io::Result<()> {
    if value.is_empty() {
        return Ok(());
    }
    let sock = UnixDatagram::unbound()?;
    sock.send_to_addr(state.as_bytes(), &socket_addr(value)?)?;
    Ok(())
}

/// Off Linux there is no systemd; every notification is a no-op.
#[cfg(not(target_os = "linux"))]
pub fn notify(_state: &str) {}

/// Sends `state` to `$NOTIFY_SOCKET` when set; failures are logged, never fatal.
#[cfg(target_os = "linux")]
pub fn notify(state: &str) {
    let Some(value) = std::env::var_os("NOTIFY_SOCKET") else {
        return;
    };
    if let Err(e) = send_to(&value.to_string_lossy(), state) {
        tracing::warn!(error = %e, state, "sd_notify failed");
    }
}

pub fn ready() {
    notify("READY=1");
}

pub fn stopping() {
    notify("STOPPING=1");
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use super::*;

    #[test]
    fn path_socket_receives_the_message() {
        let path = std::env::temp_dir().join(format!("hark-notify-{}.sock", std::process::id()));
        let _ = std::fs::remove_file(&path);
        let rx = UnixDatagram::bind(&path).unwrap();
        send_to(path.to_str().unwrap(), "READY=1").unwrap();
        let mut buf = [0u8; 64];
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"READY=1");
        send_to(path.to_str().unwrap(), "STOPPING=1").unwrap();
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"STOPPING=1");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn abstract_socket_receives_the_message() {
        let name = format!("hark-notify-test-{}", std::process::id());
        let addr = SocketAddr::from_abstract_name(name.as_bytes()).unwrap();
        let rx = UnixDatagram::bind_addr(&addr).unwrap();
        send_to(&format!("@{name}"), "READY=1").unwrap();
        let mut buf = [0u8; 64];
        let n = rx.recv(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"READY=1");
    }

    #[test]
    fn empty_value_is_ignored_and_a_dead_socket_is_an_error() {
        assert!(send_to("", "READY=1").is_ok());
        assert!(send_to("/nonexistent/hark.sock", "READY=1").is_err());
    }
}
