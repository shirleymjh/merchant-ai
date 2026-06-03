package com.yshopping.merchantai.graph;

/**
 * LangGraph 节点函数接口。
 *
 * <p>每个节点接收并返回同一个 GraphState，节点之间通过状态对象传递商家信息、
 * 意图、Doris 结果、回复话术和待落库信息。</p>
 */
@FunctionalInterface
public interface GraphNode {
    /**
     * 执行一个图节点，并返回更新后的状态。
     */
    GraphState apply(GraphState state);
}
