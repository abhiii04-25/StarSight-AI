@echo off
cd /d "%~dp0"
call ".venv\Scripts\python.exe" -m streamlit run streamlit_app.py --server.port 8512                    
