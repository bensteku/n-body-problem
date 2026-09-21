@echo off
setlocal EnableExtensions

rem Simple entry point for all supported builds.
rem Usage: build.bat
rem        build.bat legacy [Scalar|SIMD|GPU] [Full|BaHu]

if /I "%~1"=="legacy" goto legacy_args
set "BUILD_MODE=greenfield"
if not "%~1"=="" (
    echo Usage: build.bat
    exit /b 2
)
set "TARGET=nbody_app.exe"
goto args_ready

:legacy_args
set "BUILD_MODE=legacy"
set "BACKEND=%~2"
set "FORCE_MODEL=%~3"
if not "%~4"=="" (
    echo Usage: build.bat legacy [Scalar^|SIMD^|GPU] [Full^|BaHu]
    exit /b 2
)
set "TARGET=n_body_problem.exe"

:args_ready
if not defined BACKEND set "BACKEND=Scalar"
if not defined FORCE_MODEL set "FORCE_MODEL=Full"

if /I "%BACKEND%"=="Scalar" set "BACKEND=Scalar"
if /I "%BACKEND%"=="SIMD" set "BACKEND=SIMD"
if /I "%BACKEND%"=="GPU" set "BACKEND=GPU"
if /I not "%BACKEND%"=="Scalar" if /I not "%BACKEND%"=="SIMD" if /I not "%BACKEND%"=="GPU" (
    echo ERROR: backend must be Scalar, SIMD, or GPU.
    exit /b 2
)

if /I "%FORCE_MODEL%"=="Full" set "FORCE_MODEL=Full"
if /I "%FORCE_MODEL%"=="BaHu" set "FORCE_MODEL=BaHu"
if /I not "%FORCE_MODEL%"=="Full" if /I not "%FORCE_MODEL%"=="BaHu" (
    echo ERROR: force model must be Full or BaHu.
    exit /b 2
)

rem Ninja uses cl.exe directly. Always initialize the MSVC x64 environment
rem here: a Developer Shell can still contain cl.exe and headers from
rem different Visual Studio installations.
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" set "VSWHERE=%ProgramFiles%\Microsoft Visual Studio\Installer\vswhere.exe"
if exist "%VSWHERE%" for /f "usebackq delims=" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%I"
set "USER_VCPKG_ROOT=%VCPKG_ROOT%"
if defined VSINSTALL if exist "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat" >nul
if defined USER_VCPKG_ROOT set "VCPKG_ROOT=%USER_VCPKG_ROOT%"

:tools_ready
where cmake >nul 2>&1 || (
    echo ERROR: cmake was not found on PATH.
    exit /b 1
)
where ninja >nul 2>&1
if errorlevel 1 if defined VSINSTALL if exist "%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe" set "PATH=%VSINSTALL%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
set "USE_VS_GENERATOR=0"
where ninja >nul 2>&1 || set "USE_VS_GENERATOR=1"
where cl >nul 2>&1 || (
    echo ERROR: MSVC cl.exe was not found. Install the Visual C++ Build Tools.
    exit /b 1
)

set "CMAKE_TOOLCHAIN_ARG="
if defined VCPKG_ROOT (
    if not exist "%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake" (
        echo ERROR: VCPKG_ROOT does not point to a complete vcpkg checkout:
        echo        %VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake was not found.
        echo        Clone the official vcpkg repository there and run bootstrap-vcpkg.bat.
        exit /b 1
    )
    set CMAKE_TOOLCHAIN_ARG="-DCMAKE_TOOLCHAIN_FILE=%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
    echo Using vcpkg toolchain: %VCPKG_ROOT%
)

rem Keep a non-backslash final character so quoted paths are parsed correctly by CMake.
set "ROOT=%~dp0."
if /I "%BUILD_MODE%"=="legacy" (
    set "BUILD_DIR=%ROOT%\build\legacy-%BACKEND%-%FORCE_MODEL%"
) else (
    set "BUILD_DIR=%ROOT%\build\greenfield"
)
echo.
echo Building %BUILD_MODE%
echo Build directory: %BUILD_DIR%
echo.

if "%USE_VS_GENERATOR%"=="0" (
    if /I "%BUILD_MODE%"=="legacy" (
        cmake --fresh -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_MODE=%BUILD_MODE% -DBUILD_VARIANT=%BACKEND% -DFORCE_MODEL=%FORCE_MODEL% %CMAKE_TOOLCHAIN_ARG%
    ) else (
        cmake --fresh -S "%ROOT%" -B "%BUILD_DIR%" -G Ninja -DCMAKE_BUILD_TYPE=Release -DBUILD_MODE=%BUILD_MODE% %CMAKE_TOOLCHAIN_ARG%
    )
) else (
    echo Ninja was not found; using Visual Studio 2022 generator.
    if /I "%BUILD_MODE%"=="legacy" (
        cmake --fresh -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_MODE=%BUILD_MODE% -DBUILD_VARIANT=%BACKEND% -DFORCE_MODEL=%FORCE_MODEL% %CMAKE_TOOLCHAIN_ARG%
    ) else (
        cmake --fresh -S "%ROOT%" -B "%BUILD_DIR%" -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release -DBUILD_MODE=%BUILD_MODE% %CMAKE_TOOLCHAIN_ARG%
    )
)
if errorlevel 1 exit /b %errorlevel%

if "%USE_VS_GENERATOR%"=="0" (
    cmake --build "%BUILD_DIR%" --parallel
) else (
    cmake --build "%BUILD_DIR%" --config Release --parallel
)
if errorlevel 1 exit /b %errorlevel%

echo.
echo BUILD SUCCEEDED
echo Executable: %BUILD_DIR%\%TARGET%
exit /b 0
