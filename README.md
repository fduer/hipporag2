chunks_embeddings文件夹：存储的是由原生hipporag2对2wiki数据集的embedding结果

四个.graphml文件：文件名末尾带"_chunks"的是原生llama 7b生成的知识图谱与6119个chunks合并的结果。
  合并方法：①添加6119个type为passage的节点；②添加entity节点和event节点指向passage节点的边，关系为“mention in”
四个.graphml文件：文件名末尾不带"_chunks"的是原生llama 7b生成的知识图谱

注意：在.json和.graphml中，LF(\n)被替换为了CRLF(\r\n)
