local LrTasks = import "LrTasks"
local LrFileUtils = import "LrFileUtils"
local LrPathUtils = import "LrPathUtils"
local LrDialogs = import "LrDialogs"
local json = require "json"

local TaggingService = {}

function TaggingService.getTagsForImages(jsonFile)
    -- 1) Build the path to your Python script
    local scriptPath = LrPathUtils.child(_PLUGIN.path, "clip_multilabel.py")
    local pythonPath = LrPathUtils.child(_PLUGIN.path, "venv/bin/python3")

    -- 2) Choose a temporary output file (for the JSON results)
    local tempFolder = LrPathUtils.getStandardFilePath("temp")
    local outputFile = LrPathUtils.child(tempFolder, "python_output.json")

    -- 3) Construct a command that redirects stdout to outputFile
    local command = string.format(
        '"%s" "%s" "%s" %d %.2f > "%s"',
        pythonPath,
        scriptPath,
        jsonFile,
        10,     -- top_k
        0.20,   -- threshold
        outputFile
    )

    -- 4) Run the command **synchronously** (no callback)
    local exitCode = LrTasks.execute(command)

    -- If exitCode is non-zero, Python had an error
    if exitCode ~= 0 then
        LrDialogs.message("Python Error", "Non-zero exit code: " .. exitCode, "critical")
        return {}
    end

    -- 5) Read the JSON results from outputFile
    local result = LrFileUtils.readFile(outputFile)
    if not result or #result == 0 then
        LrDialogs.message("No results returned", "Check the Python script for errors.", "warning")
        return {}
    end

    LrDialogs.message("Result", result)


    -- 6) Decode the JSON into a Lua table
    local success, data = pcall(json.decode, result)
    if success and data and type(data) == "table" then
        return data
    else
        return {}
    end
end

return TaggingService