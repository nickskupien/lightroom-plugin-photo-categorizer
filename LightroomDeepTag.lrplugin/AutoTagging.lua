local LrDialogs       = import("LrDialogs")
local LrTasks         = import("LrTasks")
local LrApplication   = import("LrApplication")
local LrFileUtils     = import("LrFileUtils")
local LrPathUtils     = import("LrPathUtils")
local LrStringUtils   = import("LrStringUtils")
local json            = require("json")
local TaggingService  = require("TaggingServiceClip")

-- Helper: manually find or create a keyword by name
local function getOrCreateKeyword(catalog, name)
    -- search all existing keywords
    for _, kw in ipairs(catalog:getKeywords()) do
        if kw:getName() == name then
            return kw
        end
    end
    -- not found, so create
    return catalog:createKeyword(name, {}, false, nil)
end

local function runBatchTagging()
    local catalog = LrApplication.activeCatalog()
    local selectedPhotos = catalog:getTargetPhotos()

    if #selectedPhotos == 0 then
        LrDialogs.message("No photos selected!", "Please select photos to tag.", "info")
        return
    end

    local response = LrDialogs.confirm(
        "Run AI Auto-Tagging on " .. #selectedPhotos .. " photo(s)?",
        "This will call the Python script once in bulk."
    )

    if response == "ok" then
        LrTasks.startAsyncTask(function()
            -- 1) Gather all photo paths
            local photoPaths = {}
            for i, photo in ipairs(selectedPhotos) do
                local path = photo:getRawMetadata("path")
                table.insert(photoPaths, path)
            end

            -- 2) Create a temporary JSON file with all image paths
            local tempFolder = LrPathUtils.getStandardFilePath("temp")
            local jsonFile = LrPathUtils.child(tempFolder, "photo_paths.json")

            -- Convert photoPaths (Lua table) to JSON string
            local jsonString = json.encode(photoPaths)

            -- Write the JSON string to a temp file (using standard Lua I/O)
            local file = io.open(jsonFile, "w")
            if file then
                file:write(jsonString)
                file:close()
            else
                LrDialogs.message("Error", "Failed to open file for writing: " .. jsonFile, "critical")
                return
            end

            -- 3) Call the TaggingService to process the single JSON file
            local results = TaggingService.getTagsForImages(jsonFile)
            if not results or #results == 0 then
                LrDialogs.message("No results returned", "Check the Python script for errors.", "warning")
                return
            end

            -- Map file paths to Lightroom photo objects for quick lookup
            local pathToPhoto = {}
            for _, photo in ipairs(selectedPhotos) do
                local p = photo:getRawMetadata("path")
                pathToPhoto[p] = photo
            end

            -- 4) Process each photo in its own withWriteAccessDo
            for _, entry in ipairs(results) do
                local p = entry.image_path
                local photo = pathToPhoto[p]
                if photo and entry.tags then
                    -- One photo per transaction
                    catalog:withWriteAccessDo("Apply AI Keywords to single photo", function()
                        -- 1) Remove existing lds_ keywords
                        local existingKeywords = photo:getRawMetadata("keywords") or {}
                        for _, kwObj in ipairs(existingKeywords) do
                            local kwName = kwObj:getName()
                            if string.find(kwName, "^lds_") then
                                photo:removeKeyword(kwObj)
                            end
                        end

                        -- 2) Add new lds_ keywords
                        for _, tagName in ipairs(entry.tags) do
                            local prefixedTagName = "lds_" .. tagName
                            local kw = getOrCreateKeyword(catalog, prefixedTagName)
                            photo:addKeyword(kw)
                        end
                    end)
                end
            end

            LrDialogs.message("Batch AI Auto-Tagging Complete", "Keywords have been applied to the selected photos.", "info")
        end)
    end
end

runBatchTagging()
