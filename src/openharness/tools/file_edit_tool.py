"""String-based file editing tool. 基于字符串替换的文件编辑工具。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

"""内置工具是如何继承基类、实现execute方法、使用上下文和返回结果的，快速掌握自定义工具的写法。

AI 传入 path、old_str、new_str
系统自动校验参数格式（FileEditToolInput）
工具拿到参数 + 运行环境（context）
解析路径 → 校验安全 → 检查文件是否存在
读取内容 → 替换字符串 → 写回文件
返回 ToolResult 结果给 AI"""

"""定义 AI 传参格式"""
class FileEditToolInput(BaseModel):
    """Arguments for the file edit tool."""

    path: str = Field(description="Path of the file to edit") #要编辑的文件路径，给 AI 看描述。
    old_str: str = Field(description="Existing text to replace") #要被替换的旧字符串。
    new_str: str = Field(description="Replacement text") #替换成的新字符串。
    replace_all: bool = Field(default=False) #是否替换所有匹配项，默认只替换第一个。

"""继承 BaseTool，实现文件编辑逻辑"""
class FileEditTool(BaseTool):
    """Replace text in an existing file."""

    name = "edit_file"
    description = "Edit an existing file by replacing a string." #工具描述，给 AI 看，告诉它工具功能。
    input_model = FileEditToolInput #指定这个工具的输入参数模型（上面定义的）。

    """工具真正干活的方法"""
    # async：异步执行：arguments：AI 传来的、已经校验好的参数；context：工具运行环境（目录、钩子等）；返回值必须是 ToolResult
    async def execute(
        self,
        arguments: FileEditToolInput,
        context: ToolExecutionContext,
    ) -> ToolResult:
        #把相对路径转成绝对路径，基于上下文的当前工作目录 context.cwd
        path = _resolve_path(context.cwd, arguments.path)

        #判断当前是否在 Docker 沙箱中运行。
        from openharness.sandbox.session import is_docker_sandbox_active

        #如果在沙箱里，导入路径安全校验工具。
        if is_docker_sandbox_active():
            from openharness.sandbox.path_validator import validate_sandbox_path

            #校验路径是否安全，不安全直接返回错误结果。
            allowed, reason = validate_sandbox_path(path, context.cwd)
            if not allowed:
                return ToolResult(output=f"Sandbox: {reason}", is_error=True)

        #文件不存在 → 返回错误结果。
        if not path.exists():
            return ToolResult(output=f"File not found: {path}", is_error=True)

        #读取文件原始内容。 要替换的旧字符串不存在 → 返回错误。
        original = path.read_text(encoding="utf-8")
        if arguments.old_str not in original:
            return ToolResult(output="old_str was not found in the file", is_error=True)

        #replace_all=True → 替换所有匹配，否则 → 只替换第一个匹配
        if arguments.replace_all:
            updated = original.replace(arguments.old_str, arguments.new_str)
        else:
            updated = original.replace(arguments.old_str, arguments.new_str, 1)

        #把修改后的内容写回文件。执行成功，返回成功结果。
        path.write_text(updated, encoding="utf-8")
        return ToolResult(output=f"Updated {path}")

"""安全路径处理"""
def _resolve_path(base: Path, candidate: str) -> Path:
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()
