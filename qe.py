# mcp_qe.py
"""
MCP 工具：接入 Quantum ESPRESSO (pw.x)
功能：
- 生成 pw.x 输入文件（pwscf）
- 运行 pw.x 并捕获输出
- 解析输出（总能量、Fermi 能、收敛信息）
- 与 MCP 集成的 wrapper（返回 JSON-serializable 结果）
依赖：Python 3.7+（标准库），可选：ase 用于结构处理（若没有可使用简单 dict）
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
import json
import re

# 可选：若系统安装了 ASE，可把结构转为 pw.x 输入更方便
try:
    from ase import Atoms
    ASE_AVAILABLE = True
except Exception:
    ASE_AVAILABLE = False


# ---------------------------
# 生成 pw.x 输入（简单模板）
# ---------------------------
PW_INPUT_TEMPLATE = """&CONTROL
  calculation = '{calculation}',
  prefix = '{prefix}',
  pseudo_dir = '{pseudo_dir}',
  outdir = '{outdir}',
  tprnfor = .true.,
  tstress = .true.,
/
&SYSTEM
  ibrav = {ibrav},
  nat = {nat},
  nat_types = {nat_types},
  ntyp = {ntyp},
  ecutwfc = {ecutwfc},
  occupations = 'smearing',
  smearing = 'mp',
  degauss = {degauss},
/
&ELECTRONS
  conv_thr = {conv_thr},
  mixing_beta = {mixing_beta},
/
ATOMIC_SPECIES
{atomic_species}
CELL_PARAMETERS angstrom
{cell_parameters}
ATOMIC_POSITIONS angstrom
{atomic_positions}

K_POINTS automatic
{kpoints_grid} 0 0 0
"""

def generate_atomic_species_block(species_pseudo: Dict[str, str]) -> str:
    """
    species_pseudo: {'Fe': 'Fe.pbe-spn-rrkjus_psl.1.0.0.UPF', ...}
    返回 ATOMIC_SPECIES 部分字符串
    """
    lines = []
    for sym, pseudo in species_pseudo.items():
        # 假设质量未知，使用 symbol 占位；Quantum ESPRESSO 需要质量字段，用户可自行替换
        # 推荐：这里填写真实原子量，例如 Fe 55.845
        mass = species_pseudo.get('_masses_', {}).get(sym, 0.0)
        if mass:
            lines.append(f"{sym} {mass:.6f} {pseudo}")
        else:
            # 若未提供质量，写 1.0 占位（用户应替换为正确质量）
            lines.append(f"{sym} 1.0 {pseudo}")
    return "\n".join(lines)


def format_cell_parameters(cell: Union[list, tuple]) -> str:
    """
    cell: 3x3 list or tuple of lattice vectors (in angstrom)
    返回 3 行
    """
    return "\n".join("  {:.8f}  {:.8f}  {:.8f}".format(*vec) for vec in cell)


def format_atomic_positions(positions: list) -> str:
    """
    positions: list of tuples (symbol, x, y, z) in angstrom
    返回 ATOMIC_POSITIONS 部分
    """
    lines = []
    for sym, x, y, z in positions:
        lines.append(f"  {sym}  {x:.10f}  {y:.10f}  {z:.10f}")
    return "\n".join(lines)


def default_kpoints_from_cell(cell: Union[list, tuple]) -> Tuple[int,int,int]:
    """
    简单估算 k-points 网格，根据晶格常数最大分量
    """
    # 目标: 大约 0.2..0.3 1/angstrom resolution -> grid ~ int(10..30*a)
    import math
    lengths = [ (vec[0]**2 + vec[1]**2 + vec[2]**2)**0.5 for vec in cell ]
    max_len = max(lengths) if lengths else 10.0
    scale = max(4, int(20 / max_len))
    return (scale, scale, scale)


def generate_pw_input(
    *,
    prefix: str = "calc",
    calculation: str = "scf",
    cell: list,
    positions: list,
    species_pseudo: Dict[str, str],
    pseudo_dir: str = "./pseudo",
    outdir: str = "./tmp_out",
    ibrav: int = 0,
    nat_types: int = 0,
    ntyp: int = 0,
    ecutwfc: float = 40.0,
    degauss: float = 0.02,
    conv_thr: float = 1e-6,
    mixing_beta: float = 0.7,
    kpoints_grid: Optional[Tuple[int,int,int]] = None
) -> str:
    """
    生成 pw.x 输入文件字符串
    必填：cell (3x3 lattice vectors), positions [(symbol,x,y,z), ...], species_pseudo mapping
    """
    if kpoints_grid is None:
        kx, ky, kz = default_kpoints_from_cell(cell)
    else:
        kx, ky, kz = kpoints_grid

    atomic_species_block = generate_atomic_species_block(species_pseudo)
    cell_parameters = format_cell_parameters(cell)
    atomic_positions = format_atomic_positions(positions)

    s = PW_INPUT_TEMPLATE.format(
        calculation=calculation,
        prefix=prefix,
        pseudo_dir=pseudo_dir,
        outdir=outdir,
        ibrav=ibrav,
        nat=len(positions),
        nat_types=nat_types if nat_types else len({p[0] for p in positions}),
        ntyp=ntyp if ntyp else len({p[0] for p in positions}),
        ecutwfc=ecutwfc,
        degauss=degauss,
        conv_thr=conv_thr,
        mixing_beta=mixing_beta,
        atomic_species=atomic_species_block,
        cell_parameters=cell_parameters,
        atomic_positions=atomic_positions,
        kpoints_grid=f"{kx} {ky} {kz}"
    )
    return s


# ---------------------------
# 运行 pw.x 并捕获输出
# ---------------------------
def run_pw(
    pw_input_str: str,
    pw_cmd: str = "pw.x",
    workdir: Optional[Union[str, Path]] = None,
    input_filename: str = "pw_input.in",
    output_filename: str = "pw_output.out",
    timeout: Optional[int] = None,
) -> Dict[str, Any]:
    """
    在 workdir 中运行 pw.x
    返回： {'returncode': int, 'stdout': str, 'stderr': str, 'output_path': Path }
    """
    if workdir is None:
        workdir = Path(tempfile.mkdtemp(prefix="mcp_qe_"))
    else:
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

    in_path = workdir / input_filename
    out_path = workdir / output_filename

    # 写入输入文件
    in_path.write_text(pw_input_str, encoding='utf-8')

    # 运行 pw.x
    # pw.x < input > output
    cmd = [pw_cmd]
    # many systems expect STDIN -> pw.x
    # We'll run as: pw.x -in pw_input.in OR pw.x < pw_input.in
    # prefer redirect via shell-like redirection with subprocess
    try:
        with in_path.open('rb') as fin, out_path.open('wb') as fout:
            proc = subprocess.run(
                cmd,
                stdin=fin,
                stdout=fout,
                stderr=subprocess.PIPE,
                cwd=str(workdir),
                timeout=timeout
            )
            stderr = proc.stderr.decode('utf-8', errors='ignore') if proc.stderr else ""
            retval = {
                "returncode": proc.returncode,
                "stderr": stderr,
                "output_path": str(out_path),
                "workdir": str(workdir),
            }
            # 读取部分 stdout 供快速查看
            try:
                # 只取最后 2000 字节作为快速预览
                raw = out_path.read_bytes()
                preview = raw[-20000:].decode('utf-8', errors='ignore')
            except Exception:
                preview = ""
            retval["preview"] = preview
            return retval
    except FileNotFoundError as e:
        raise RuntimeError(f"pw.x 可执行文件未找到：{pw_cmd}. 错误：{e}")
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"pw.x 运行超时: {e}")


# ---------------------------
# 解析 pw.x 输出（简单）
# ---------------------------
def parse_pw_output_basic(output_text: str) -> Dict[str, Any]:
    """
    从 pw.x 输出文本中解析关键信息：
    - total energy (!) lines like: !    total energy              =   -11.90420956 Ry
    - Fermi energy: 'the Fermi energy is' or 'Fermi energy is'
    - convergence info: 'convergence has been achieved' / 'converged' or final forces
    返回字典
    """
    res = {}
    # 总能量（Ry）
    m = re.search(r"!\s*total energy\s*=\s*([-\d\.Eed+]+)\s*Ry", output_text, re.IGNORECASE)
    if m:
        try:
            res['total_energy_ry'] = float(m.group(1))
        except:
            pass

    # 也有可能的不同格式
    if 'total_energy_ry' not in res:
        m2 = re.search(r"total-energy\s*=\s*([-\d\.Eed+]+)\s*Ry", output_text, re.IGNORECASE)
        if m2:
            try:
                res['total_energy_ry'] = float(m2.group(1))
            except:
                pass

    # 费米能（eV）
    m = re.search(r"the Fermi energy is\s*([-\d\.Eed+]+)\s*eV", output_text, re.IGNORECASE)
    if m:
        try:
            res['fermi_energy_ev'] = float(m.group(1))
        except:
            pass
    else:
        # 有时显示为 'Fermi energy is'
        m2 = re.search(r"Fermi energy is\s*([-\d\.Eed+]+)\s*eV", output_text, re.IGNORECASE)
        if m2:
            try:
                res['fermi_energy_ev'] = float(m2.group(1))
            except:
                pass

    # 收敛信息
    if re.search(r"convergence has been achieved", output_text, re.IGNORECASE):
        res['converged'] = True
    elif re.search(r"not converged", output_text, re.IGNORECASE):
        res['converged'] = False

    # Forces/Stress summary（若有）
    # 查找最后的 "Total force" 或类似行
    forces = re.findall(r"total force\s*=\s*\(?\s*([-\d\.\sEe,+]+)\)?", output_text, re.IGNORECASE)
    if forces:
        res['forces_summary'] = forces[-1].strip()

    # 记录最后若干行作日志
    res['last_lines'] = "\n".join(output_text.strip().splitlines()[-40:])

    return res


def parse_pw_output_from_file(output_path: Union[str, Path]) -> Dict[str, Any]:
    txt = Path(output_path).read_text(encoding='utf-8', errors='ignore')
    return parse_pw_output_basic(txt)


# ---------------------------
# MCP 风格封装接口
# ---------------------------
def run_qe_job(
    *,
    structure: Optional[Union[Atoms, Dict]] = None,
    cell: Optional[list] = None,
    positions: Optional[list] = None,
    species_pseudo: Optional[Dict[str,str]] = None,
    pw_cmd: str = "pw.x",
    workdir: Optional[str] = None,
    ecutwfc: float = 40.0,
    kpoints: Optional[Tuple[int,int,int]] = None,
    additional_ctrl: Optional[Dict[str,str]] = None,
    timeout: Optional[int] = 3600
) -> Dict[str,Any]:
    """
    高层接口：接受 ASE Atoms 或手动指定 cell/positions，运行 pw.x 并解析结果。
    返回字典，适合直接返回给 MCP 调用者。
    """
    # 支持 ASE 对象作为结构输入
    if ASE_AVAILABLE and structure is not None and isinstance(structure, Atoms):
        ase_atoms: Atoms = structure
        cell = ase_atoms.get_cell().tolist()
        positions = [(sym, *pos.tolist()) for sym, pos in zip(ase_atoms.get_chemical_symbols(), ase_atoms.get_positions())]
        # species pseudo 需要用户提供
    else:
        # 若直接传入 dict 结构
        if isinstance(structure, dict) and structure:
            # 允许 structure = {'cell':..., 'positions': [...]}
            if 'cell' in structure and 'positions' in structure:
                cell = cell or structure['cell']
                positions = positions or structure['positions']

    if cell is None or positions is None or species_pseudo is None:
        raise ValueError("必须提供 cell, positions, species_pseudo（赝势文件名映射）")

    # prepare workdir
    if workdir:
        workdir_path = Path(workdir)
        workdir_path.mkdir(parents=True, exist_ok=True)
    else:
        workdir_path = Path(tempfile.mkdtemp(prefix="mcp_qe_job_"))

    outdir = str((workdir_path / "qe_out").absolute())

    # generate input
    prefix = additional_ctrl.get('prefix', 'mcp_calc') if additional_ctrl else 'mcp_calc'
    pw_input = generate_pw_input(
        prefix=prefix,
        calculation=additional_ctrl.get('calculation', 'scf') if additional_ctrl else 'scf',
        cell=cell,
        positions=positions,
        species_pseudo=species_pseudo,
        pseudo_dir=additional_ctrl.get('pseudo_dir', str(Path(workdir_path)/"pseudo")),
        outdir=outdir,
        ecutwfc=ecutwfc,
        kpoints_grid=kpoints
    )

    # optionally copy pseudo files into pseudo_dir if provided as absolute paths mapping
    # species_pseudo map value can be absolute path or filename (assumed in pseudo_dir)
    pseudo_dir_path = Path(additional_ctrl.get('pseudo_dir', str(Path(workdir_path)/"pseudo")))
    if not pseudo_dir_path.exists():
        pseudo_dir_path.mkdir(parents=True, exist_ok=True)
    for sym, upf in species_pseudo.items():
        if sym.startswith('_'):  # skip meta keys
            continue
        upf_path = Path(upf)
        if upf_path.exists():
            # copy into pseudo_dir
            shutil.copy(upf_path, pseudo_dir_path / upf_path.name)
            species_pseudo[sym] = upf_path.name  # update mapping to filename

    # write input file
    input_file = workdir_path / "pw_input.in"
    input_file.write_text(pw_input, encoding='utf-8')

    # run pw.x
    result = run_pw(
        pw_input_str=pw_input,
        pw_cmd=pw_cmd,
        workdir=workdir_path,
        input_filename=input_file.name,
        output_filename="pw_output.out",
        timeout=timeout
    )

    # parse output
    parsed = {}
    try:
        parsed = parse_pw_output_from_file(result['output_path'])
    except Exception as e:
        parsed['parse_error'] = str(e)

    # prepare final return dict
    ret = {
        "workdir": str(workdir_path),
        "input_file": str(input_file),
        "output_file": result.get('output_path'),
        "pw_returncode": result.get('returncode'),
        "pw_stderr": result.get('stderr'),
        "pw_preview": result.get('preview'),
        "parsed": parsed
    }
    return ret


# ---------------------------
# MCP 注册示例（伪代码）
# ---------------------------
# 下面示例展示如何在 MCP 中将该函数作为工具注册/调用。
# 具体根据你们 MCP 框架的注册 API 调整。
#
# from mcp.server.fastmcp import FastMCP
# mcp = FastMCP(...)
#
# def mcp_qe_tool_handler(task):
#     # task 可以包含 structure, species_pseudo, etc.
#     out = run_qe_job(
#         structure=task['structure'],
#         species_pseudo=task['species_pseudo'],
#         pw_cmd=task.get('pw_cmd', 'pw.x'),
#         additional_ctrl=task.get('additional_ctrl', {}),
#         kpoints=task.get('kpoints', None),
#         timeout=task.get('timeout', 3600)
#     )
#     return out
#
# # 注册示例（伪代码）
# mcp.register_tool("quantum_espresso_run", mcp_qe_tool_handler)
