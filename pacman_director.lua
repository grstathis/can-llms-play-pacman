-- pacman_director.lua: freeze-frame via savestate reload (non-blocking)
-- Mac version with WRAM dump for score/lives tracking.
--
-- Mesen kills Lua callbacks that run longer than ~1s, so we never busy-wait.
-- While "frozen", the same savestate is reloaded every frame and the command
-- file is polled once per frame.
-- Mesen only allows create/loadSavestate inside an exec memory callback, so
-- we register a one-shot exec callback whenever one is needed.
local scriptData = emu.getScriptDataFolder()
local OUTPUT_DIR = scriptData .. "/output/"
os.execute('mkdir -p "' .. OUTPUT_DIR .. '"')

local cmdFile = scriptData .. "/command.txt"
local statusFile = OUTPUT_DIR .. "status.txt"

local frame = 0
local turnCount = 0
local BOOT_WAIT = 30 -- frames for the loaded save state to settle

local S_BOOT, S_PLAYING, S_FROZEN = 1, 2, 3
local state = S_BOOT

local moveDir = nil
local moveFrames = 60
local movePlayed = 0

local frozenState = nil
local frozenReady = false
local pendingAction = nil
local cbRef = nil
local ADDR_START, ADDR_END = 0x000000, 0xFFFFFF

local function makeInput(dir)
    if dir == "up" then return {up = true}
    elseif dir == "down" then return {down = true}
    elseif dir == "left" then return {left = true}
    elseif dir == "right" then return {right = true}
    end
    return nil
end

local function screenshot()
    local png = emu.takeScreenshot()
    if png then
        local f = io.open(OUTPUT_DIR .. "screen.png", "wb")
        if f then f:write(png); f:close() end
    end
end

-- Dump work RAM so the Python side can read lives/score.
-- Written before status "waiting", so it always matches the screenshot.
local WRAM_DUMP_SIZE = 0x4000
local function dumpWram()
    local t = {}
    for i = 0, WRAM_DUMP_SIZE - 1 do
        t[#t + 1] = string.char(emu.read(i, emu.memType.snesWorkRam))
    end
    local f = io.open(OUTPUT_DIR .. "wram.bin", "wb")
    if f then f:write(table.concat(t)); f:close() end
end

local function writeStatus(msg)
    local f = io.open(statusFile, "w")
    if f then f:write(msg .. "\n"); f:close() end
end

local function clearCommand()
    local f = io.open(cmdFile, "w")
    if f then f:write("wait\n"); f:close() end
end

local function readCommand()
    local f = io.open(cmdFile, "r")
    if not f then return nil end
    local content = f:read("*a")
    f:close()
    if not content or not content:find("\n") then return nil end
    if content:match("^quit") then return "quit", 0 end
    local dir, fr = content:match("^(%S+)%s+(%d+)")
    if dir and fr and dir ~= "wait" then return dir, tonumber(fr) end
    return nil
end

local function onExec()
    local action = pendingAction
    pendingAction = nil
    if cbRef then
        emu.removeMemoryCallback(cbRef, emu.callbackType.exec, ADDR_START, ADDR_END)
        cbRef = nil
    end
    if action == "create" then
        frozenState = emu.createSavestate()
        frozenReady = true
        writeStatus("waiting")
    elseif action == "load" and frozenState then
        emu.loadSavestate(frozenState)
    end
end

local function requestAction(action)
    pendingAction = action
    if not cbRef then
        cbRef = emu.addMemoryCallback(onExec, emu.callbackType.exec, ADDR_START, ADDR_END)
    end
end

local function freeze()
    screenshot()
    dumpWram()
    clearCommand()
    frozenReady = false
    state = S_FROZEN
    requestAction("create")
end

clearCommand()
writeStatus("booting")

emu.addEventCallback(function()
    if state == S_PLAYING then
        local input = makeInput(moveDir)
        if input then emu.setInput(input, 0) end
    else
        emu.setInput({}, 0)
    end
end, emu.eventType.inputPolled)

emu.addEventCallback(function()
    frame = frame + 1

    if state == S_BOOT then
        if frame >= BOOT_WAIT then
            emu.log("Save state settled - freezing for AI")
            freeze()
        end

    elseif state == S_PLAYING then
        movePlayed = movePlayed + 1
        if movePlayed >= moveFrames then
            turnCount = turnCount + 1
            emu.log("Turn " .. turnCount .. " complete - freezing for AI")
            freeze()
        end

    elseif state == S_FROZEN then
        if not frozenReady then return end

        local dir, fr = readCommand()
        if dir == "quit" then emu.stop(0); return end
        if dir then
            moveDir = dir
            moveFrames = fr
            movePlayed = 0
            writeStatus("playing " .. dir .. " " .. fr)
            emu.log("Move: " .. dir .. " " .. fr)
            state = S_PLAYING
        end
        requestAction("load")
    end
end, emu.eventType.endFrame)

emu.log("=== PACMAN DIRECTOR (Mac + WRAM Dump) ===")
