-- This file calls the Python CLIP script for local AI tagging
local CLIP = {}

function CLIP.run(imagePath)
    local command = "venv/bin/python3 clip_tagging_3.py " .. imagePath
    local handle = io.popen(command)
    local result = handle:read("*a")
    handle:close()
    return result
end

return CLIP
