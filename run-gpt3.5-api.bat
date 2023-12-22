@echo off



set SCRIPT_NAME=SocketServer.py
set CHATVER=3
set STREAM=True
set CHARACTER=catmaid
set MODEL=gpt-3.5-turbo
set KEY=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

python %SCRIPT_NAME% --chatVer %CHATVER% --stream %STREAM% --character %CHARACTER% --model %MODEL% --APIKey %KEY%
