@echo off
for %%f in (*.json) do (
    echo Processing %%f
    labelme_export_json "%%f"
)