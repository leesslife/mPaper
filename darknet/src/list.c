#include <stdlib.h>
#include <string.h>
#include "list.h"

list *make_list()
{
	list *l = malloc(sizeof(list));
	l->size = 0;
	l->front = 0;
	l->back = 0;
	return l;
} //以上创建一个空的node链表，size代表链表长度，front代表链表首节点，back代表末节点这里都是空的

/*
void transfer_node(list *s, list *d, node *n)
{
    node *prev, *next;
    prev = n->prev;
    next = n->next;
    if(prev) prev->next = next;
    if(next) next->prev = prev;
    --s->size;
    if(s->front == n) s->front = next;
    if(s->back == n) s->back = prev;
}
*/

void *list_pop(list *l){
    if(!l->back) return 0;
    node *b = l->back;
    void *val = b->val;
    l->back = b->prev;
    if(l->back) l->back->next = 0;
    free(b);
    --l->size;
    
    return val;
}

void list_insert(list *l, void *val)          //链表插入操作？
{
	node *new = malloc(sizeof(node));         //建立一个node指针 分配空间
	/*typedef struct node{
    	void *val;
    	struct node *next;
    	struct node *prev;
    } node; //node节点，next指向下一个节点 prev指向前一个节点*/
	new->val = val;                           //将val指针指向 val指向的内容
	new->next = 0;                            //node的next指向空

	if(!l->back){                             //如果list l指针的尾指针不存在，进行以下操作
		l->front = new;                       //l->back如果是空的，说明list对应的链表中没有其他节点
		new->prev = 0;                        //将new node节点前后都清空 l->front前指针指向new指向的内容
	}else{                                    //如果l->back指针是非空的，说明list对应的链表中本就存在node节点，可以执行以下操作
		l->back->next = new;				  //首先将l->back指向的末节点 node的next指针指向new节点
		new->prev = l->back;                  //然后new->prev指向原来末尾指针，完成双向链表的插入操作
	}
	l->back = new;                            //最后list *l指针指向的list 中的back指针指向new，完成list 对于双向链表的首尾指向操作
	++l->size;                                //list->size的大小加1
}

void free_node(node *n)
{
	node *next;
	while(n) {
		next = n->next;
		free(n);
		n = next;
	}
}

void free_list(list *l)
{
	free_node(l->front);
	free(l);
}

void free_list_contents(list *l)
{
	node *n = l->front;
	while(n){
		free(n->val);
		n = n->next;
	}
}

void **list_to_array(list *l)
{
    void **a = calloc(l->size, sizeof(void*));
    int count = 0;
    node *n = l->front;
    while(n){
        a[count++] = n->val;
        n = n->next;
    }
    return a;
}
