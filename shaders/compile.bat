@echo off
setlocal

pushd "%~dp0" >nul
slangc shader.slang -target spirv -profile spirv_1_4 -emit-spirv-directly -fvk-use-entrypoint-name -entry vertMain -entry fragMain -o slang.spv
set "exit_code=%ERRORLEVEL%"
popd >nul

exit /b %exit_code%
