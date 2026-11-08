#!/bin/bash
# Helper: send a move and wait for the next screenshot
# Usage: ./play_helper.sh <direction> <frames>
DIR="$1"
FRAMES="${2:-60}"

SCRIPT_DATA=~/Library/Application\ Support/MesenCE/LuaScriptData/pacman_director
CMD="$SCRIPT_DATA/command.txt"
STATUS="$SCRIPT_DATA/output/status.txt"
SCREEN="$SCRIPT_DATA/output/screen.png"
WRAM="$SCRIPT_DATA/output/wram.bin"
LOCAL=/Users/sgrigoropoul/ogre-battle-python/tools/mesen/pacman_mac/output/screen.png

# Record screen mtime before move
PREV_MTIME=$(stat -f %m "$SCREEN" 2>/dev/null || echo 0)

# Send the move
echo "$DIR $FRAMES" > "$CMD"

# Wait for new screenshot + waiting status
for i in $(seq 1 120); do
    CUR_MTIME=$(stat -f %m "$SCREEN" 2>/dev/null || echo 0)
    CUR_STATUS=$(cat "$STATUS" 2>/dev/null)
    if [ "$CUR_MTIME" != "$PREV_MTIME" ] && [ "$CUR_STATUS" = "waiting" ]; then
        cp "$SCREEN" "$LOCAL"
        # Read WRAM for score/lives
        if [ -f "$WRAM" ]; then
            # Lives at 0x028C, Score at 0x02A2 (3 bytes BCD)
            LIVES=$(xxd -s 0x028C -l 1 -p "$WRAM" 2>/dev/null)
            SCORE_B0=$(xxd -s 0x02A2 -l 1 -p "$WRAM" 2>/dev/null)
            SCORE_B1=$(xxd -s 0x02A3 -l 1 -p "$WRAM" 2>/dev/null)
            SCORE_B2=$(xxd -s 0x02A4 -l 1 -p "$WRAM" 2>/dev/null)
            echo "READY lives=0x${LIVES} score_bcd=${SCORE_B2}${SCORE_B1}${SCORE_B0}"
        else
            echo "READY (no wram)"
        fi
        exit 0
    fi
    sleep 0.05
done
echo "TIMEOUT status=$(cat "$STATUS" 2>/dev/null)"
exit 1
