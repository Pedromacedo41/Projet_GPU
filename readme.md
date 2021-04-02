

# Batch sort - Trifusion

   - Authors:  Pedro Macedo Flores and Hudson Braga Vieira 


## Usage
### Compilation

 ```shell
 $ nvcc *.cu -o projet
 ```


 ### Command line 

```shell
$ ./projet d batch_dim
if d <= 1024 and batch_dim = 1, the program will ask you if you want to use the proc's shared memory: Answer y/n.
```



- **d:**  dimension of each vector of batch
- **batch_dim:** batch_dim - batch dimension


## File description 

- **main.cu:** main file 
- **utils.cu:** random vector generation and other useful functions


