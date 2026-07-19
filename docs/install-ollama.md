# Installing Ollama

Ollama is the local model runtime this kit is built on. Install it once, pull a
model, and the server runs in the background on `http://localhost:11434`.

## Windows (native)

1. Download the installer from <https://ollama.com/download> and run it. It
   installs a background service and the `ollama` command.
2. Open a new PowerShell window and verify:

   ```powershell
   ollama --version
   ollama pull llama3.1:8b
   ollama run llama3.1:8b "Say hello in one sentence."
   ```

3. The service starts automatically on login. If the API is not responding, start
   it manually:

   ```powershell
   ollama serve
   ```

GPU acceleration on Windows uses your NVIDIA driver directly; no CUDA toolkit
install is required. Make sure your driver is current via GeForce Experience or
the NVIDIA site.

## Windows with WSL2

Running inside WSL2 is useful if the rest of your toolchain is Linux. GPU
passthrough works with a recent NVIDIA driver on the Windows host.

1. Install or update WSL2 and a distro:

   ```powershell
   wsl --install
   wsl --update
   ```

2. Inside the WSL2 shell, install Ollama:

   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull llama3.1:8b
   ```

3. Confirm the GPU is visible from WSL2 with `nvidia-smi`. If it lists your GPU,
   Ollama will use it. You do not install a separate driver inside WSL2; the
   Windows host driver provides the CUDA runtime.

Note: a native Windows Ollama install and a WSL2 install are separate servers.
Pick one for a given project so the model store and the port do not collide.

## macOS

1. Download the app from <https://ollama.com/download> or install via Homebrew:

   ```bash
   brew install ollama
   ```

2. Start the server and pull a model:

   ```bash
   ollama serve        # or launch the menu-bar app
   ollama pull llama3.1:8b
   ```

On Apple Silicon, Ollama uses the Metal backend and unified memory automatically.
An M-series Mac with 16 GB or more runs 8B models comfortably.

## Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.1:8b
```

The install script sets up a systemd service. Manage it with
`systemctl status ollama` and `journalctl -u ollama` for logs. For NVIDIA GPUs,
install the proprietary driver; Ollama bundles the CUDA runtime it needs.

## Verifying from this kit

Once Ollama is running:

```bash
python -m src.model_manager list
python -m src.model_manager recommend
```

If `list` reports it cannot reach the server, start it with `ollama serve` and
confirm nothing else is bound to port 11434.

## Changing the host or port

Set `OLLAMA_HOST` for the server, and the same variable in your `.env` so the kit
knows where to look:

```bash
# serve on all interfaces, custom port
OLLAMA_HOST=0.0.0.0:11500 ollama serve
```

```dotenv
# .env for the kit
OLLAMA_HOST=http://localhost:11500
```
