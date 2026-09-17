@echo off
setlocal EnableExtensions

rem Simple entry point for all supported builds.
rem Usage: build.bat [Scalar|SIMD|CUDA] [Full|BaHu]
rem        build.bat legacy [Scalar|SIMD|CUDA] [Full|BaHu]

if /I "%~1"=="legacy" goto legacy_args
set "BUILD_MODE=greenfield"
set "BACKEND=%~1"
set "FORCE_MODEL=%~2"
set "TARGET=nbody_app.exe"
goto args_ready

:legacy_args
set "BUILD_MODE=legacy"
set "BACKEND=%~2"
set "FORCE_MODEL=%~3"
set "TARGET=n_body_problem.exe"

:args_ready
if not defined BACKEND set "BACKEND=Scalar"
if not defined FORCE_MODEL set "FORCE_MODEL=Full"

if /I "%BACKEND%"=="Scalar" set "BACKEND=Scalar"
if /I "%BACKEND%"=="SIMD" set "BACKEND=SIMD"
if /I "%BACKEND%"=="CUDA" set "BACKEND=CUDA"
if /I not "%BACKEND%"=="Scalar" if /I not "%BACKEND%"=="SIMD" if /I not "%BACKEND%"=="CUDA" (
    echo ERROR: backend must be Scalar, SIMD, or CUDA.
    exit /b 2
)

if /I "%FORCE_MODEL%"=="Full" set "FORCE_MODEL=Full"
if /I "%FORCE_MODEL%"=="BaHu" set "FORCE_MODEL=BaHu"
if /I not "%FORCE_MODEL%"=="Full" if /I not "%FORCE_MODEL%"=="BaHu" (
    echo ERROR: force model must be Full or BaHu.
    exit /b 2
)

rem Ninja uses cl.exe directly, so initialize the MSVC x64 environment when needed.
where cl >nul 2>&1
if not errorlevel 1 goto tools_ready

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
if defined VSINSTALL if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul

:tools_ready
where cmake >nul 2>&1 || (
    echo ERROR: cmake was not found on PATH.
    exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 if defined VSINSTALL if exist "%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
where ninja >nul 2>&1 || (
    echo ERROR: ninja was not found on PATH or in Visual Studio.
    exit /b 1
)
where cl >nul 2>&1 || (
    echo ERROR: MSVC cl.exe was not found. Install the Visual C++ Build Tools.
    exit /b 1
)

rem Keep a non-backslash final character so quoted paths are parsed correctly by CMake.
set "ROOT=%~dp0."
set "BUILD_DIR=%ROOT%\build\%BUILD_MODE%-%BACKEND%-%FORCE_MODEL%"
echo.
echo Building %BUILD_MODE% / %BACKEND% / %FORCE_MODEL%
echo Build directory: %BUILD_DIR%
echo.

cmake -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_MODE=%BUILD_MODE% -DBUILD_VARIANT=%BACKEND% -DFORCE_MODEL=%FORCE_MODEL%
if errorlevel 1 exit /b %errorlevel%

cmake --build "%BUILD_DIR%" --parallel
if errorlevel 1 exit /b %errorlevel%

echo.
echo BUILD SUCCEEDED
echo Executable: %BUILD_DIR%\%TARGET%
exit /b 0
