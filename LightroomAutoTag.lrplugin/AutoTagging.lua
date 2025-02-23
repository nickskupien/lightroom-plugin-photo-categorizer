local LrDialogs = import 'LrDialogs'
local LrTasks = import 'LrTasks'
local LrApplication = import 'LrApplication'
local LrFileUtils = import 'LrFileUtils'
local LrPathUtils = import 'LrPathUtils'
local LrStringUtils = import 'LrStringUtils'
local json = require 'json'
local TaggingService = require 'TaggingService'

-- Removes all tags that start with "lds_"
local function removeLdsTags(keywordString)
    if not keywordString or keywordString == "" then
        return ""
    end

    -- Split the comma-separated string
    local remainingTags = {}
    for tag in string.gmatch(keywordString, "([^,]+)") do
        local trimmed = LrStringUtils.trimWhitespace(tag)
        -- Keep the tag if it does NOT start with "lds_"
        if not string.find(trimmed, "^lds_") then
            table.insert(remainingTags, trimmed)
        end
    end

    -- Rebuild a comma-separated string
    return table.concat(remainingTags, ", ")
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

            -- Write the JSON string to a temp file
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

            -- 4) Write tags back to each photo in a single write operation
            catalog:withWriteAccessDo("Apply AI Tags in Bulk", function()
                -- 'results' is an array of { image_path = "...", tags = { {tag, score}, ... } }
                local pathToPhoto = {}
                for _, photo in ipairs(selectedPhotos) do
                    local p = photo:getRawMetadata("path")
                    pathToPhoto[p] = photo
                end

                for _, entry in ipairs(results) do
                    local p = entry.image_path
                    local photo = pathToPhoto[p]
                    if photo and entry.tags then
                        -- 1) Remove existing lds_ tags
                        local existing = photo:getFormattedMetadata("keywordTags") or ""
                        local existingWithoutLds = removeLdsTags(existing)

                        -- 2) Build new keywords from existing (without lds_)
                        local newKeywords = existingWithoutLds
                        for _, tagData in ipairs(entry.tags) do
                            local tagName = tagData[1]
                            local confidence = tagData[2]
                            if confidence > 0.21 then
                                -- Prepend the "lds_" prefix
                                local prefixedTagName = "lds_" .. tagName

                                -- If we already have some keywords, add a comma
                                if newKeywords ~= "" then
                                    newKeywords = newKeywords .. ", " .. prefixedTagName
                                else
                                    newKeywords = prefixedTagName
                                end
                            end
                        end

                        -- 3) Write the updated keyword string back to the photo
                        photo:setRawMetadata("keywordTags", newKeywords)
                    end
                end
            end)

            LrDialogs.message("Batch AI Auto-Tagging Complete", "Tags have been applied to the selected photos.", "info")
        end)
    end
end

runBatchTagging()
