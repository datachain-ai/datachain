import os
import subprocess
import sys
from pathlib import Path

# ConfiguraÃ§Ã£o, ajuste apenas se necessÃ¡rio.

# DiretÃ³rio raiz do projeto clonado, os testes vai comeÃ§ar a execuÃ§Ã£o a petir dele.
PROJETO = "."

# DiretÃ³rio dos testes detectado automaticamente, mas pode forÃ§ar manualmente
# Exemplos: TESTES = "./tests"  ou  TESTES = "./test"
TESTES = None

# Pasta onde os relatÃ³rios serÃ£o salvos (nÃ£o altere)
PASTA = "metrics-after-pytest"

# DetecÃ§Ã£o automÃ¡tica do diretÃ³rio de testes
CANDIDATOS = ["tests", "test", "src/tests", "src/test"]

if TESTES is None:
    for candidato in CANDIDATOS:
        if Path(candidato).exists():
            TESTES = candidato
            break

if TESTES is None:
    print("Erro: diretÃ³rio de testes nÃ£o encontrado.")
    print(f"Procurado em: {CANDIDATOS}")
    print("Defina manualmente a variÃ¡vel TESTES no script.")
    sys.exit(1)

# ExecuÃ§Ã£o
os.makedirs(PASTA, exist_ok=True)

print(f"Projeto : {os.path.abspath(PROJETO)}")
print(f"Testes  : {TESTES}")
print(f"RelatÃ³rios em: {PASTA}/")
print()

resultado = subprocess.run(
    [
        sys.executable,
        "-m",
        "pytest",
        TESTES,
        "-v",
        f"--junit-xml={os.path.join(PASTA, 'pytest_antes.xml')}",
        f"--html={os.path.join(PASTA, 'pytest_antes.html')}",
        "--self-contained-html",
        f"--cov={PROJETO}",
        "--cov-branch",
        f"--cov-report=xml:{os.path.join(PASTA, 'coverage_antes.xml')}",
        f"--cov-report=json:{os.path.join(PASTA, 'coverage_antes.json')}",
        f"--cov-report=html:{os.path.join(PASTA, 'coverage_antes_html')}",
        "--cov-report=term-missing",
    ],
    cwd=PROJETO,
    text=True,
    encoding="utf-8",
)

print(f"\nExit code: {resultado.returncode}")
print(f"\nArquivos gerados em '{PASTA}':")
print("  pytest_depois.xml      â†’ resultados dos testes em XML")
print("  pytest_depois.html     â†’ relatÃ³rio visual dos testes")
print("  coverage_depois.xml    â†’ cobertura de cÃ³digo em XML")
print("  coverage_depois.json   â†’ cobertura de cÃ³digo em JSON")
print("  coverage_depois_html/  â†’ relatÃ³rio visual de cobertura")
print("\nConcluÃ­do.")
