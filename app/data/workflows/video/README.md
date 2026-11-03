# Video workflows (ComfyUI API format)

Drop video graphs exported from ComfyUI via **Save (API Format)** here as
`<name>.json`. Enable in `~/.config/soul-of-waifu/media.yaml`
(`video_enabled: true`).

Optional node mapping file `<name>.meta.yaml`: same keys —
`text_node_id`, `output_node_id` (VHS_VideoCombine). Auto-detection:
PrimitiveString/CLIPTextEncode → VHS_VideoCombine/SaveImage.
