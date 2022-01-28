program -> analysis -> fuse -> tensorize match -> schedule
bmm + softmax + bmm -> bmm + exp + bmm + div

o
|
o  o
| /
x
|
o  o
| /
o
|
o  o
| /
x
|
o