# home-manager module for voiced.
#
# It lives here rather than in the consuming config so that a change to the
# CLI or the config schema and the change to how the unit launches it land
# in one commit.
#
# Import it as `inputs.voiced.homeModules.default` and set policy only:
#
#   imports = [ inputs.voiced.homeModules.default ];
#   services.voiced.enable = true;
{ self }:
{ lib, config, pkgs, ... }:

let
  cfg = config.services.voiced;

  defaultPackage = self.packages.${pkgs.stdenv.hostPlatform.system}.default;

  tomlFormat = pkgs.formats.toml { };

  args = lib.escapeShellArgs ([ "start" ] ++ lib.optional cfg.http "--http" ++ cfg.extraArgs);
in
{
  options.services.voiced = {
    enable = lib.mkEnableOption "voiced — speech-to-text and text-to-speech daemon";

    package = lib.mkOption {
      type = lib.types.package;
      default = defaultPackage;
      defaultText = lib.literalExpression "inputs.voiced.packages.\${system}.default";
      description = "The voiced package to install and run.";
    };

    http = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Start the HTTP server alongside the Unix socket, so other programs
        on the machine (or the LAN, depending on `settings.server.host`)
        can reach the STT and TTS endpoints. Without it voiced answers
        only on its local socket.
      '';
    };

    startAtLogin = lib.mkOption {
      type = lib.types.bool;
      default = true;
      description = ''
        Pull the unit in at login. Set false when something else owns the
        lifecycle -- a compositor `exec-once`, or a hand-run
        `systemctl --user start voiced`. Worth turning off on a machine
        where you would rather not hold the STT and TTS models in VRAM
        from login onward; `unload_timeout_minutes` releases them only
        after the first load.
      '';
    };

    cudaDevice = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "0";
      example = null;
      description = ''
        Value of `CUDA_VISIBLE_DEVICES` for the unit. Null leaves the
        variable unset, which lets the inference worker see every GPU.
      '';
    };

    waylandDisplay = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = "wayland-1";
      description = ''
        Value of `WAYLAND_DISPLAY` for the unit. Desktop mode injects
        transcribed text with `wl-copy`, which needs a compositor socket,
        and a systemd user unit does not inherit one. Null leaves it unset
        for a headless host that only serves the HTTP API.
      '';
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [ ];
      description = "Additional arguments appended to `voiced start`.";
    };

    extraEnvironment = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = { };
      example = lib.literalExpression ''{ VOICED_LOG_LEVEL = "debug"; }'';
      description = "Extra environment variables for the unit.";
    };

    settings = lib.mkOption {
      inherit (tomlFormat) type;
      default = { };
      example = lib.literalExpression ''
        {
          unload_timeout_minutes = 60;
          server.host = "0.0.0.0";
          transcription.replacements."cloud code" = "Claude Code";
        }
      '';
      description = ''
        Contents of `~/.config/voiced/config.toml`. Empty by default and no
        file is written, which leaves voiced's own defaults in charge --
        they live in `src/voiced/config.py` and are the only copy. Set the
        keys you want to differ, not a full file: a restated default is a
        default that stops tracking upstream the moment upstream changes
        it, and nothing reports the divergence.

        `server.host` defaults to loopback upstream. voiced has no
        authentication, so a non-loopback value opens transcription and
        synthesis to everything that can route to this host.
      '';
    };
  };

  config = lib.mkIf cfg.enable {
    home.packages = [
      cfg.package
      # voiced shells out to these by name rather than bundling them:
      # injector.py calls `wl-copy` for clipboard delivery, and `wtype`
      # types the text into the focused window.
      pkgs.wtype
      pkgs.wl-clipboard
    ];

    xdg.configFile = lib.mkIf (cfg.settings != { }) {
      "voiced/config.toml".source = tomlFormat.generate "voiced-config.toml" cfg.settings;
    };

    systemd.user.services.voiced = {
      Unit = {
        Description = "voiced — speech-to-text and text-to-speech daemon";
        After = [ "network.target" ];
      };

      Service = {
        Type = "simple";
        ExecStart = "${cfg.package}/bin/voiced ${args}";
        Restart = "on-failure";
        RestartSec = 5;
        Environment = lib.mapAttrsToList (k: v: "${k}=${v}") (
          lib.optionalAttrs (cfg.cudaDevice != null) { CUDA_VISIBLE_DEVICES = cfg.cudaDevice; }
          // lib.optionalAttrs (cfg.waylandDisplay != null) { WAYLAND_DISPLAY = cfg.waylandDisplay; }
          // {
            # %t is the user runtime dir (/run/user/<uid>), so the unit does
            # not hardcode a uid.
            XDG_RUNTIME_DIR = "%t";
          }
          // cfg.extraEnvironment
        );
      };
    }
    # Plain `//`, not lib.mkIf: systemd.user.services values are opaque to
    # the module system, so a mkIf buried inside one is written to the unit
    # file verbatim instead of being resolved.
    // lib.optionalAttrs cfg.startAtLogin {
      Install.WantedBy = [ "default.target" ];
    };
  };
}
