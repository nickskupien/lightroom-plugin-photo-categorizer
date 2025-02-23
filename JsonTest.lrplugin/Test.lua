local LrDialogs = import("LrDialogs")
local json      = require("json")  -- We'll test loading json.lua

local data = { foo = "bar", nums = {1, 2, 3} }
local encoded = json.encode(data)
LrDialogs.message("JSON Test", "Encoded: " .. encoded, "info")
