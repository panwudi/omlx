class FlytoMlx < Formula
  desc "Flyto MLX — Apple Silicon LLM server with audio chat, DFlash, and Chinese model presets (fork of oMLX)"
  homepage "https://github.com/panwudi/flyto-mlx"
  url "https://github.com/panwudi/flyto-mlx/archive/refs/tags/v0.4.1.tar.gz"
  sha256 "0f0a671c371e9d3061eb09d6c61ed56db75557477d10039ed51c56bed158b094"
  license "Apache-2.0"

  head "https://github.com/panwudi/flyto-mlx.git", branch: "main"

  option "with-grammar", "Install xgrammar for structured output (requires torch, ~2GB)"

  depends_on "rust" => :build
  depends_on "python@3.11"
  depends_on :macos
  depends_on arch: :arm64

  # mlx-audio pins mlx-lm==0.31.1 which conflicts with our git-pinned mlx-lm.
  # Fetch source separately so we can patch the pin before install.
  resource "mlx-audio" do
    url "https://github.com/Blaizzy/mlx-audio.git",
      revision: "51753266e0a4f766fd5e6fbc46652224efc23981"
  end

  service do
    run [opt_bin/"fmlx", "serve"]
    keep_alive true
    working_dir var
    log_path var/"log/flyto-mlx.log"
    error_log_path var/"log/flyto-mlx.log"
    environment_variables PATH: std_service_path_env
  end

  def install
    # Create venv with pip so dependency resolution works properly
    system "python3.11", "-m", "venv", libexec

    # Build Rust-based packages from source with headerpad to prevent
    # Homebrew dylib ID fixup failure (Mach-O header too small for absolute paths).
    ENV.append "LDFLAGS", "-Wl,-headerpad_max_install_names"

    # Install flyto-mlx (with optional grammar extra for structured output)
    install_spec = build.with?("grammar") ? "#{buildpath}[grammar]" : buildpath.to_s
    system libexec/"bin/pip", "install", "--no-binary", "pydantic-core,rpds-py,tiktoken", install_spec

    # Install mlx-audio with patched mlx-lm pin to avoid version conflict
    resource("mlx-audio").stage do
      inreplace "pyproject.toml", '"mlx-lm==0.31.1"', '"mlx-lm>=0.31.1"'
      system libexec/"bin/pip", "install", ".[all]"
    end

    # python-multipart is declared in flyto-mlx's [audio] extra, not in mlx-audio
    system libexec/"bin/pip", "install", "python-multipart>=0.0.5"

    # Install both CLI names:
    #   fmlx — Flyto MLX brand primary
    #   omlx — backward-compat alias for users migrating from upstream oMLX
    bin.install_symlink Dir[libexec/"bin/fmlx"]
    bin.install_symlink Dir[libexec/"bin/omlx"]
  end

  # Patch the macOS arm64 xgrammar wheel so its native binding loads.
  # See upstream commentary (jundot/omlx#1005) for the underlying issue —
  # we inherit the same workaround unchanged.
  def post_install
    return unless build.with?("grammar")

    ohai "Patching xgrammar macOS arm64 wheel"
    py = libexec/"bin/python"
    site = Utils.safe_popen_read(py, "-c",
                                 "import site; print(site.getsitepackages()[0])").chomp
    tvmlib = Utils.safe_popen_read(py, "-c",
      "import os, tvm_ffi; print(os.path.join(os.path.dirname(tvm_ffi.__file__), 'lib'))").chomp
    dylib = "#{site}/xgrammar/libxgrammar_bindings.dylib"
    dist_dirs = Dir["#{site}/xgrammar-*.dist-info"]

    ohai "  site=#{site}"
    ohai "  tvmlib=#{tvmlib}"
    ohai "  dylib=#{dylib} (exists? #{File.exist?(dylib)})"
    ohai "  dist-info=#{dist_dirs.inspect}"

    odie "xgrammar dylib not found at #{dylib}" unless File.exist?(dylib)
    odie "xgrammar dist-info not found under #{site}" if dist_dirs.empty?

    rpaths = Utils.safe_popen_read("/usr/bin/otool", "-l", dylib)
    if rpaths.include?(tvmlib)
      ohai "  rpath already points at tvm_ffi/lib"
    else
      ohai "  adding rpath -> #{tvmlib}"
      system "/usr/bin/install_name_tool", "-add_rpath", tvmlib, dylib
      system "/usr/bin/codesign", "--force", "--sign", "-", dylib
    end

    record = "#{dist_dirs.first}/RECORD"
    if File.exist?(record) && File.read(record).include?("libxgrammar_bindings.dylib")
      ohai "  RECORD already lists the dylib"
    else
      ohai "  writing dylib entry to #{record}"
      File.open(record, "a") { |f| f.puts "xgrammar/libxgrammar_bindings.dylib,," }
    end

    ohai "  verifying import xgrammar..."
    system py, "-c", "import xgrammar; print('xgrammar import OK')"
  end

  test do
    assert_match version.to_s, shell_output("#{bin}/fmlx --version")
  end
end
