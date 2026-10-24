//! T025: the `casos` subcommand accepts the battery and rejects a broken case.

use std::path::PathBuf;
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_ds41-paridad"))
}

fn cases_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("casos")
}

#[test]
fn the_case_battery_is_valid_and_exits_zero() {
    let output = Command::new(binary())
        .args(["casos", "--casos"])
        .arg(cases_dir())
        .output()
        .expect("run ds41-paridad casos");
    assert!(
        output.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("casos:"));
}

#[test]
fn a_broken_case_exits_one_with_the_reason() {
    let dir = std::env::temp_dir().join(format!("ds41-paridad-casos-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("temp dir");
    std::fs::write(
        dir.join("roto.json"),
        r#"{"id":"roto","protocol":"chat","body":{},"expect":{"render":"igual"}}"#,
    )
    .expect("write case");
    let output = Command::new(binary())
        .args(["casos", "--casos"])
        .arg(&dir)
        .output()
        .expect("run ds41-paridad casos");
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(output.status.code(), Some(1));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("expect.render igual"), "stderr: {stderr}");
}
