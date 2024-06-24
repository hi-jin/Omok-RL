PORT=8000
SCREEN_MODE=real

cd omok-game
nohup pipenv run python main.py --port=$PORT --screen-mode=$SCREEN_MODE > ../game.log 2>&1 & echo $! > ../game.pid
cd ..

cd omok-rl 
nohup pipenv run python agent.py --server-port=$PORT --who=black --mode=test --model-path=./models/black/model.zip > ../agent-black.log 2>&1 & echo $! > ../agent-black.pid
nohup pipenv run python agent.py --server-port=$PORT --who=white --mode=test --model-path=./models/white/model.zip > ../agent-white.log 2>&1 & echo $! > ../agent-white.pid  
cd ..
