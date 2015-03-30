#include<stdio.h>
#include<stdlib.h>
#include<pthread.h>

#ifndef  MYLIST_H
#define  MYLIST_H

typedef struct _myListElem {
    _myListElem *prev;
    _myListElem *next;
    unsigned int val;
} myListElem;

class myList {
      
public:
  myList(){
    _head = NULL;
    _tail = NULL;
    _nElem = 0;
    pthread_mutex_init(&_mutex, NULL);
  }
  
  ~myList(){
    clear();
    pthread_mutex_destroy(&_mutex); 
  }
      
  void clear(){
    if( size() > 0 ){
      myListElem *cur = _head;

      pthread_mutex_lock(&_mutex);
      while( cur != NULL ){
	free(cur->prev);
	cur = cur->next;
      }
      free(cur);
      _nElem = 0;
      _head  = NULL;
      _tail  = NULL;
      pthread_mutex_unlock(&_mutex);
    }
  }

  void push_back(unsigned int val){
    myListElem *cur = (myListElem *)malloc(sizeof(myListElem));
    if( NULL == cur ){
      fprintf(stderr,"myList: could not allocate RAM\n");
      jallocation_error("myList: could not allocate RAM\n");
    }
    cur->prev = NULL;
    cur->next = NULL;
    cur->val  = val;

    if( NULL == _head ){
      pthread_mutex_lock(&_mutex);
      _head = cur;
      _tail = cur;
      pthread_mutex_unlock(&_mutex);
    }
    else{
      cur->prev   = _tail;
      _tail->next = cur;
      _tail = cur;
    }
    incrNElem();
  }

  void pop_front(){
    if( NULL == _head ) {
      fprintf(stderr,"myList::pop_front() : there is no component\n");
      _nElem = 0;
      return;
    }
    myListElem *newHead = _head->next;
    free(_head);
    _head = newHead;
    _head->prev = NULL;
    decrNElem();
  }
  
  unsigned int back(){
    if( NULL == _tail ) {
      fprintf(stderr,"myList::back() : there is no component\n");
      jallocation_error(  "myList::back() : there is no component\n");
    }
    return _tail->val;
  }
  
  unsigned int front(){
    if( NULL == _head ) {
      fprintf(stderr,"myList::front() : there is no component\n");
      jallocation_error(  "myList::front() : there is no component\n");
    }
    return _head->val;
  }
      
  size_t size(){
    return _nElem;
  }

 private:
  void incrNElem(){
    pthread_mutex_lock(&_mutex);
    _nElem++;
    pthread_mutex_unlock(&_mutex);
  }
  void decrNElem(){
    pthread_mutex_lock(&_mutex);
    _nElem--;
    pthread_mutex_unlock(&_mutex);
  }

 private:
  myListElem *_head;
  myListElem *_tail; 
  size_t      _nElem; /* Note that this variable should be thread-safe */
  pthread_mutex_t _mutex;
};


#endif
