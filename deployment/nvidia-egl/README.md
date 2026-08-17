# NVIDIA EGL for headless Blender

`10_nvidia.json` selects `libEGL_nvidia.so.0` through GLVND so Blender EEVEE can
render on an NVIDIA GPU without an X11/Wayland desktop.

`deployment/start_services.sh` exports this project-local vendor file only
when `/dev/nvidiactl` and the NVIDIA EGL library are both available. GPU-less
instances keep the system EGL selection.

The system EGL selection remains unchanged when the NVIDIA device or library is
not available.
