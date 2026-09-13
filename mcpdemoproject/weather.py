from mcp.server.fastmcp import FastMCP
mcp=FastMCP("Weather")# server name

@mcp.tool()
async def get_weather(location:str)->str:
    """ get the weather location"""
    return "Its always raining in california"
#The transport="stdio" argument tells the server to:
#Use standard input/output (stdio and stdout) to receive and respond to tool function calls


if __name__=="__main__":
    mcp.run(transport="streamable-http")
