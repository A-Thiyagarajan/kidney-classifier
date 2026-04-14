@echo off
echo Activating .venv...
call .venv\Scripts\activate.bat
if errorlevel 1 (
  echo Failed to activate .venv
  pause
  exit /b 1
)
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements-fixed.txt
if errorlevel 1 (
  echo Install failed
  pause
  exit /b 1
)
echo Testing TensorFlow...
python check.py
echo Testing model...
python test_model.py
echo Setup complete! Run: python app.py
pause

