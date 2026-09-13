from mcp.server.fastmcp import FastMCP
mcp=FastMCP("Math")# server name

@mcp.tool()
def add(a:int,b:int)->int:
    """_summary_
    Add two numbers
    """
    return a+b

@mcp.tool()
def multiple(a:int,b:int)->int:
    """_summary_
    Multiply two numbers
    """
    return a*b
#The transport="stdio" argument tells the server to:
#Use standard input/output (stdio and stdout) to receive and respond to tool function calls


if __name__=="__main__":
    mcp.run(transport="stdio")
