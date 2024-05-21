chcp 65001

@echo off

:: endnum 结束数字
:: format 文件后缀

echo 要创建多少文件：
set /p endnum=

echo 要创建什么格式：
echo 1-.sql
echo 2-.java
echo 3-.txt
echo 请输入：
set /p format=

if "%format%"=="1" (
for /l %%i in (1, 1, %endnum%) do (
echo. >%%i.sql
)
)

if "%format%"=="2" (
for /l %%i in (1, 1, %endnum%) do (
echo. >%%i.java
)
)

if "%format%"=="3" (
for /l %%i in (0, 1, %endnum%) do (
echo. >output_%%i.txt
)
)

echo 创建完成
pause