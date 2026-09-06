@echo off
setlocal

set VERSION=20260723
set BASE_URL=https://d2ej7fkh96fzlu.cloudfront.net/sudachidict-raw

cd /d "%~dp0"

echo Downloading small_lex.zip...
curl -fL "%BASE_URL%/%VERSION%/small_lex.zip" -o small_lex.zip
if errorlevel 1 exit /b 1

echo Downloading core_lex.zip...
curl -fL "%BASE_URL%/%VERSION%/core_lex.zip" -o core_lex.zip
if errorlevel 1 exit /b 1

echo Downloading matrix.def.zip...
curl -fL "%BASE_URL%/matrix.def.zip" -o matrix.def.zip
if errorlevel 1 exit /b 1

echo Extracting...

tar -xf small_lex.zip
if errorlevel 1 exit /b 1

tar -xf core_lex.zip
if errorlevel 1 exit /b 1

tar -xf matrix.def.zip
if errorlevel 1 exit /b 1

del /q small_lex.zip
del /q core_lex.zip
del /q matrix.def.zip

echo Done.
endlocal