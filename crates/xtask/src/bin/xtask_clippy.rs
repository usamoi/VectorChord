// This software is licensed under a dual license model:
//
// GNU Affero General Public License v3 (AGPLv3): You may use, modify, and
// distribute this software under the terms of the AGPLv3.
//
// Elastic License v2 (ELv2): You may also use, modify, and distribute this
// software under the Elastic License v2, which has specific restrictions.
//
// We welcome any commercial collaboration or support. For inquiries
// regarding the licenses, please contact us at:
// vectorchord-inquiry@tensorchord.ai
//
// Copyright (c) 2025 TensorChord Inc.

use clap::Parser;
use std::collections::HashMap;
use std::env::var_os;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = target_triple::TARGET, env = "TARGET")]
    target: String,
    #[arg(long, default_value = "release", env = "PROFILE")]
    profile: String,
}

fn pg_config(pg_config: impl AsRef<Path>) -> Result<HashMap<String, String>, Box<dyn Error>> {
    let mut command = Command::new(pg_config.as_ref());
    command.stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_output = command.output()?;
    let contents = String::from_utf8(command_output.stdout)?;
    let mut result = HashMap::new();
    for line in contents.lines() {
        if let Some((key, value)) = line.split_once(" = ") {
            result.insert(key.to_string(), value.to_string());
        }
    }
    Ok(result)
}

fn clippy(
    pg_config: impl AsRef<Path>,
    pg_version: &str,
    profile: &str,
    target: &str,
) -> Result<(), Box<dyn Error>> {
    let mut command = Command::new("cargo");
    command
        .args(["clippy"])
        .args(["--workspace"])
        .args(["--profile", profile])
        .args(["--target", target])
        .args(["--features", pg_version])
        .env("PGRX_PG_CONFIG_PATH", pg_config.as_ref())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit());
    eprintln!("Running {command:?}");
    let command_status = command.spawn()?.wait()?;
    if !command_status.success() {
        return Err(format!("Cargo clippy failed: {command_status}").into());
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();
    if !std::fs::exists("./vchord.control")? {
        return Err("The script must be run from the VectorChord source directory.".into());
    }
    let Cli { target, profile } = cli;
    let path = if let Some(value) = var_os("PGRX_PG_CONFIG_PATH") {
        PathBuf::from(value)
    } else {
        return Err("Environment variable `PGRX_PG_CONFIG_PATH` is not set.".into());
    };
    let pg_config = pg_config(&path)?;
    let pg_version = {
        let version = pg_config["VERSION"].clone();
        if let Some(prefix_stripped) = version.strip_prefix("PostgreSQL ") {
            if let Some((stripped, _)) = prefix_stripped.split_once(|c: char| !c.is_ascii_digit()) {
                format!("pg{stripped}",)
            } else {
                format!("pg{prefix_stripped}",)
            }
        } else {
            return Err("PostgreSQL version is invalid.".into());
        }
    };
    clippy(&path, &pg_version, &profile, &target)?;
    Ok(())
}
