@echo off
rem Abre a bancada cnx: sobe o servidor local e abre o navegador. Feche esta janela para encerrar.
cd /d "%~dp0.."
start "" "http://127.0.0.1:8765"
node tools\bancada_server.js
