//! `ds41-paridad`: compare the gateway's DeepSeek API rendering with TensorFold's
//! token route, and check the end-to-end API shape against a scripted engine.

use std::path::PathBuf;

use clap::{Parser, Subcommand};
use ds41_gateway::paridad;

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "1" | "true" | "TRUE" | "yes" => Ok(true),
        "0" | "false" | "FALSE" | "no" => Ok(false),
        other => Err(format!("expected 1/0/true/false, got {other}")),
    }
}

#[derive(Debug, Parser)]
#[command(name = "ds41-paridad", about = "DeepSeek API parity comparator")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Validate the case files (unique ids, fields, expectations).
    Casos {
        /// Directory with `*.json` cases.
        #[arg(long, default_value = "gateway/casos")]
        casos: PathBuf,
    },
    /// Compare the API render with the Zig `/tokenize` render.
    Render {
        #[arg(long, default_value = "gateway/casos")]
        casos: PathBuf,
        #[arg(long)]
        zig: String,
        #[arg(long)]
        tokenizer: PathBuf,
        #[arg(long)]
        gateway: Option<String>,
        #[arg(long, default_value = "DeepSeek-V4.1-Flash-1017")]
        served_name: String,
        #[arg(long, num_args = 0..=1, default_missing_value = "true", value_parser = parse_bool)]
        default_thinking: Option<bool>,
        #[arg(long = "filtro")]
        filtro: Vec<String>,
        #[arg(long)]
        salida: PathBuf,
        #[arg(long)]
        motor_dir: Option<PathBuf>,
    },
    /// Check the non-stream and stream API, the invariants and the token hashes.
    E2e {
        #[arg(long, default_value = "gateway/casos")]
        casos: PathBuf,
        #[arg(long)]
        gateway: String,
        #[arg(long)]
        zig: String,
        #[arg(long, default_value = "/served/tokenizer.json")]
        tokenizer: PathBuf,
        #[arg(long, default_value = "DeepSeek-V4.1-Flash-1017")]
        served_name: String,
        #[arg(long, default_value_t = 2)]
        paralelo: usize,
        #[arg(long = "filtro")]
        filtro: Vec<String>,
        #[arg(long)]
        salida: PathBuf,
        #[arg(long)]
        motor_dir: Option<PathBuf>,
    },
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let result = match cli.command {
        Command::Casos { casos } => paridad::run_casos(&casos),
        Command::Render {
            casos,
            zig,
            tokenizer,
            gateway,
            served_name,
            default_thinking,
            filtro,
            salida,
            motor_dir,
        } => {
            let motor_dir = motor_dir.or_else(paridad::default_motor_dir);
            paridad::run_render(
                &casos,
                &filtro,
                &zig,
                gateway.as_deref(),
                &tokenizer,
                &served_name,
                default_thinking.unwrap_or(true),
                &salida,
                motor_dir.as_deref(),
            )
            .await
        }
        Command::E2e {
            casos,
            gateway,
            zig,
            tokenizer,
            served_name,
            paralelo,
            filtro,
            salida,
            motor_dir,
        } => {
            let motor_dir = motor_dir.or_else(paridad::default_motor_dir);
            paridad::run_e2e(
                &casos,
                &filtro,
                &gateway,
                &zig,
                &tokenizer,
                &served_name,
                paralelo,
                &salida,
                motor_dir.as_deref(),
            )
            .await
        }
    };
    match result {
        Ok(code) => std::process::exit(code),
        Err(error) => {
            eprintln!("ds41-paridad: {error}");
            std::process::exit(2);
        }
    }
}
