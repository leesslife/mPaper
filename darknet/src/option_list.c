#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "option_list.h"
#include "utils.h"

list *read_data_cfg(char *filename)
{
    FILE *file = fopen(filename, "r");
    if(file == 0) file_error(filename);
    char *line;
    int nu = 0;
    list *options = make_list();
    while((line=fgetl(file)) != 0){
        ++ nu;
        strip(line);
        switch(line[0]){
            case '\0':
            case '#':
            case ';':
                free(line);
                break;
            default:
                if(!read_option(line, options)){
                    fprintf(stderr, "Config file error line %d, could parse: %s\n", nu, line);
                    free(line);
                }
                break;
        }
    }
    fclose(file);
    return options;
}

metadata get_metadata(char *file)
{
    metadata m = {0};
    list *options = read_data_cfg(file);

    char *name_list = option_find_str(options, "names", 0);
    if(!name_list) name_list = option_find_str(options, "labels", 0);
    if(!name_list) {
        fprintf(stderr, "No names or labels found\n");
    } else {
        m.names = get_labels(name_list);
    }
    m.classes = option_find_int(options, "classes", 2);
    free_list(options);
    return m;
}

int read_option(char *s, list *options)
{
    size_t i;                        //系统int 32位unsigned int 64位的话是long unsigned int                
    size_t len = strlen(s);          //对应的字符串长度
    char *val = 0;                   //新建字符串，val初始值为0，就是指空指针
    for(i = 0; i < len; ++i){        //遍历整个字符串
        if(s[i] == '='){             //找到"=",
            s[i] = '\0';             //将等号变为字符串结束语"\0"
            val = s+i+1;             //将val跳转到"="下一步的位置
            break;                   //跳出当前循环
        }
    }                                
    if(i == len-1) return 0;         //i==len-1指代没有找到"=" return 0
    char *key = s;                   //key代表 =号前面的字符串
    //val代表等号后面的字符串
    option_insert(options, key, val);
    //往options的node双向链表中 插入node节点，node->val 指向kvp kvp->key=key ,kvp->val=val,kvp->used=0默认值。
    return 1;
}

void option_insert(list *l, char *key, char *val)
{
    kvp *p = malloc(sizeof(kvp));  
    /*typedef struct{
        char *key;
        char *val;
        int used;
    } kvp; //一个结构体，key代表键的字符串，val代表指的字符串，used？*/
    p->key = key;
    p->val = val;
    p->used = 0;
    //p->used=0 代表未被使用？
    list_insert(l, p);
    //将p插入(list)l指针指向的node双向链表中，node->val 指向p
}

void option_unused(list *l)
{
    node *n = l->front;
    while(n){
        kvp *p = (kvp *)n->val;
        if(!p->used){
            fprintf(stderr, "Unused field: '%s = %s'\n", p->key, p->val);
        }
        n = n->next;
    }
}

char *option_find(list *l, char *key)
{
    node *n = l->front;
    //找到node双向链表的头节点，node *n
    while(n){
        kvp *p = (kvp *)n->val;  //将节点中的void *val指向的kvp结构取出
        /*
        typedef struct{
            char *key;
            char *val;
            int used;
        } kvp; //一个结构体，key代表键的字符串，val代表指的字符串，used？*/
        if(strcmp(p->key, key) == 0){
            //strcmp(str1,str2)，若str1=str2，则返回零；若str1<str2，则返回负数；若str1>str2，则返回正数
            //两个字符串自左向右逐个字符相比（按ASCII值大小相比较），直到出现不同的字符或遇'\0'为止
            p->used = 1;
            //p->used 赋值1 表明当前参数已经倍使用过了
            return p->val;
            //返回key对应的val值
        }
        n = n->next;
    }
    return 0;
}
char *option_find_str(list *l, char *key, char *def)
{
    char *v = option_find(l, key);
    if(v) return v;
    if(def) fprintf(stderr, "%s: Using default '%s'\n", key, def);
    return def;
}

int option_find_int(list *l, char *key, int def)
{
    char *v = option_find(l, key);
    //将node双向链表以及key传入，周到key对应的val，并返回
    if(v) return atoi(v);
    //atoi 一个char *v字符串变成整形，并返回
    fprintf(stderr, "%s: Using default '%d'\n", key, def);
    //如果没有找到则打印参数并，返回def 默认值
    return def;
}

int option_find_int_quiet(list *l, char *key, int def) //这个函数更option_find_int 完全相同但它不会输出任何信息
{
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;
}

float option_find_float_quiet(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;
}

float option_find_float(list *l, char *key, float def)
{
    char *v = option_find(l, key);
    //同option_find_int
    if(v) return atof(v);
     //同option_find_int 但这里将char * 字符串转化成float
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
     //同option_find_int,但这里返回一个float
    return def;
}
