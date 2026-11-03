# Image workflows (ComfyUI API format)

Drop any txt2img/img2img graph exported from ComfyUI via **Save (API
Format)** here as `<name>.json` — it becomes an image-generation preset
automatically.

Optional node mapping file `<name>.meta.yaml`:

```yaml
text_node_id: "6"        # positive-prompt text node
output_node_id: "9"      # SaveImage node
```

Without a meta file the loader auto-detects PrimitiveString/CLIPTextEncode
for the prompt and SaveImage for output. Files are re-scanned on every
generation — changes apply immediately.
