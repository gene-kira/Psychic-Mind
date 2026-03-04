@echo off

:: ============================
::  Silent Windows Loader
::  Runs automatically after installation
:: ============================

set NEWUSER=Operator

echo Creating local admin account "%NEWUSER%"...
net user "%NEWUSER%" * /add
net localgroup administrators "%NEWUSER%" /add
net user "%NEWUSER%" /active:yes
net user "%NEWUSER%" /expires:never

echo Disabling built-in Administrator...
net user "Administrator" /active:no

echo Removing OOBE temporary account...
net user "defaultUser0" /delete >nul 2>&1

echo Applying OOBE registry modifications...
set OOBE_KEY=HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\OOBE

reg delete "%OOBE_KEY%" /v DefaultAccountAction   /f >nul 2>&1
reg delete "%OOBE_KEY%" /v DefaultAccountSAMName  /f >nul 2>&1
reg delete "%OOBE_KEY%" /v DefaultAccountSID      /f >nul 2>&1
reg delete "%OOBE_KEY%" /v LaunchUserOOBE         /f >nul 2>&1

reg add "%OOBE_KEY%" /v SkipMachineOOBE /t REG_DWORD /d 1 /f >nul 2>&1

echo Loader complete. Rebooting...
shutdown /r /t 0



    put it here  \Sources\$OEM$\$$\Setup\Scripts\SetupComplete.cmd   