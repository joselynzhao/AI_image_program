#include <iostream>
#include <math.h>
#include <fstream>
#include <cassert>
#include <string>
using namespace std;


template <class T>
int getArrayLen(T&array)
{
    return sizeof(array) / sizeof(array[0]);
}



struct number_positon{
	int number;
	int x;
	int y;
};

int h(int N,int current_state[][6],int goal_state[][6]){  // N是边长，错位计算，返回总的错位个数。 
	int count = 0;
	for(int i=0;i<N;i++){
		for(int  j = 0;j<N;j++){  //遍历每一个位置
			if(current_state[i][j]!=goal_state[i][j]){
				count++;
			}
		}
	}
	return count;
}

number_positon * get_number_position(int N,int state[][6]){  //最好返回一个排好序的list
	number_positon np_list[N*N]; 
	// count = 0;
	for(int i = 0;i<N;i++){
		for(int j = 0; j< N;j++){
			number_positon np;
			np.number = state[i][j];
			np.x = i;
			np.y = j;
			np_list[np.number]=np;
			// count++;
		}
	}
	return np_list;
}

// int h2(N,current_state[][],goal_state[][]){ // N是边长，计算曼哈顿距离之和
// 	int sum = 0;
// 	//首先得构造number_position的一个list
// 	current_np_list = get_number_position(N,current_state);
// 	goal_np_list = get_number_position(N,goal_state);
// 	for(int i=1;i<N*N;i++){  //这里不考虑空白块
// 		bias = abs(current_np_list[i].x - goal_np_list[i].x) + abs(current_np_list[i].y - goal_np_list[i].y);
// 		printf("number %d bias = %d\n",i,bias);
// 		sum +=bias;
// 	}
// 	return sum;
// }


int main(){
	int n,init_arr[36],goal_arr[36];
	ifstream infile;
	infile.open("npuzzle_in.txt",ios::in);
	infile>>n;
	printf("%d\n",n );
	for(int i=0;i<n*n;i++){
    	infile>>init_arr[i];
    	printf("%d ",init_arr[i]);
	}
	printf("\n");
	for(int i=0;i<n*n;i++){
		infile>>goal_arr[i];
		printf("%d ",goal_arr[i]);
	}


	// printf("%s\n", "hhh");
	// ifstream infile; 
	// string file = "npuzzle_in.txt";
 //    infile.open(file.data());   //将文件流对象与文件连接起来 
 //    // assert(infile.is_open());   //若失败,则输出错误消息,并终止程序运行 
 //    char c;
 //    while (!infile.eof())
 //    {
 //    	printf("%s\n","hellozhaojing" );
 //        infile >> c;
 //        cout<<c<<endl;

 //    }
 //    infile.close();             //关闭文件输入流 

	return 0;
}