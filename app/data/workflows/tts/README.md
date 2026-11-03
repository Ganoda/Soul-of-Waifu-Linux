# TTS workflows (ComfyUI API format)

Drop any workflow exported from ComfyUI via **Save (API Format)** here as
`<name>.json` — it becomes a selectable TTS engine preset automatically.

Optional node mapping file `<name>.meta.yaml`:

```yaml
text_node_id: "3"        # node receiving the spoken text
output_node_id: "9"      # SaveAudio node
```

Without a meta file the loader auto-detects: first PrimitiveString /
CLIPTextEncode / *TTS* node for input; first SaveAudio / VHS_VideoCombine
node for output. Files are re-scanned on every generation — changes apply
immediately.
