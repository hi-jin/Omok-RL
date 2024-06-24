PORT=8000
SCREEN_MODE=virtual

cd omok-game
nohup pipenv run python main.py --port=$PORT --screen-mode=$SCREEN_MODE > ../game.log 2>&1 & echo $! > ../game.pid
cd ..

cd omok-rl 
nohup pipenv run python agent.py --server-port=$PORT --who=black > ../agent-black.log 2>&1 & echo $! > ../agent-black.pid
nohup pipenv run python agent.py --server-port=$PORT --who=white > ../agent-white.log 2>&1 & echo $! > ../agent-white.pid  
cd ..


# Get the script name
script_name=$(basename "$0")

# Kill the processes started by the script
kill_script=$(cat << EOF
#!/bin/bash

# Kill processes started by $script_name
kill -9 \$(cat *.pid) 2>/dev/null
rm *.pid 2>/dev/null
EOF
)

echo "$kill_script" > "${script_name%.*}.kill"
chmod +x "${script_name%.*}.kill"

echo "Kill script created: ${script_name%.*}.kill"
