Ŝ#
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
?
ResizeBilinear
images"T
size
resized_images"
Ttype:
2	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
stack0_enc0_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestack0_enc0_conv0/kernel
?
,stack0_enc0_conv0/kernel/Read/ReadVariableOpReadVariableOpstack0_enc0_conv0/kernel*&
_output_shapes
:*
dtype0
?
stack0_enc0_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestack0_enc0_conv0/bias
}
*stack0_enc0_conv0/bias/Read/ReadVariableOpReadVariableOpstack0_enc0_conv0/bias*
_output_shapes
:*
dtype0
?
stack0_enc0_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestack0_enc0_conv1/kernel
?
,stack0_enc0_conv1/kernel/Read/ReadVariableOpReadVariableOpstack0_enc0_conv1/kernel*&
_output_shapes
:*
dtype0
?
stack0_enc0_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestack0_enc0_conv1/bias
}
*stack0_enc0_conv1/bias/Read/ReadVariableOpReadVariableOpstack0_enc0_conv1/bias*
_output_shapes
:*
dtype0
?
stack0_enc1_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestack0_enc1_conv0/kernel
?
,stack0_enc1_conv0/kernel/Read/ReadVariableOpReadVariableOpstack0_enc1_conv0/kernel*&
_output_shapes
:*
dtype0
?
stack0_enc1_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestack0_enc1_conv0/bias
}
*stack0_enc1_conv0/bias/Read/ReadVariableOpReadVariableOpstack0_enc1_conv0/bias*
_output_shapes
:*
dtype0
?
stack0_enc1_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*)
shared_namestack0_enc1_conv1/kernel
?
,stack0_enc1_conv1/kernel/Read/ReadVariableOpReadVariableOpstack0_enc1_conv1/kernel*&
_output_shapes
:*
dtype0
?
stack0_enc1_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*'
shared_namestack0_enc1_conv1/bias
}
*stack0_enc1_conv1/bias/Read/ReadVariableOpReadVariableOpstack0_enc1_conv1/bias*
_output_shapes
:*
dtype0
?
stack0_enc2_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*)
shared_namestack0_enc2_conv0/kernel
?
,stack0_enc2_conv0/kernel/Read/ReadVariableOpReadVariableOpstack0_enc2_conv0/kernel*&
_output_shapes
:$*
dtype0
?
stack0_enc2_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*'
shared_namestack0_enc2_conv0/bias
}
*stack0_enc2_conv0/bias/Read/ReadVariableOpReadVariableOpstack0_enc2_conv0/bias*
_output_shapes
:$*
dtype0
?
stack0_enc2_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$$*)
shared_namestack0_enc2_conv1/kernel
?
,stack0_enc2_conv1/kernel/Read/ReadVariableOpReadVariableOpstack0_enc2_conv1/kernel*&
_output_shapes
:$$*
dtype0
?
stack0_enc2_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*'
shared_namestack0_enc2_conv1/bias
}
*stack0_enc2_conv1/bias/Read/ReadVariableOpReadVariableOpstack0_enc2_conv1/bias*
_output_shapes
:$*
dtype0
?
stack0_enc3_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$6*)
shared_namestack0_enc3_conv0/kernel
?
,stack0_enc3_conv0/kernel/Read/ReadVariableOpReadVariableOpstack0_enc3_conv0/kernel*&
_output_shapes
:$6*
dtype0
?
stack0_enc3_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*'
shared_namestack0_enc3_conv0/bias
}
*stack0_enc3_conv0/bias/Read/ReadVariableOpReadVariableOpstack0_enc3_conv0/bias*
_output_shapes
:6*
dtype0
?
stack0_enc3_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:66*)
shared_namestack0_enc3_conv1/kernel
?
,stack0_enc3_conv1/kernel/Read/ReadVariableOpReadVariableOpstack0_enc3_conv1/kernel*&
_output_shapes
:66*
dtype0
?
stack0_enc3_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*'
shared_namestack0_enc3_conv1/bias
}
*stack0_enc3_conv1/bias/Read/ReadVariableOpReadVariableOpstack0_enc3_conv1/bias*
_output_shapes
:6*
dtype0
?
&stack0_enc5_middle_expand_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:6Q*7
shared_name(&stack0_enc5_middle_expand_conv0/kernel
?
:stack0_enc5_middle_expand_conv0/kernel/Read/ReadVariableOpReadVariableOp&stack0_enc5_middle_expand_conv0/kernel*&
_output_shapes
:6Q*
dtype0
?
$stack0_enc5_middle_expand_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*5
shared_name&$stack0_enc5_middle_expand_conv0/bias
?
8stack0_enc5_middle_expand_conv0/bias/Read/ReadVariableOpReadVariableOp$stack0_enc5_middle_expand_conv0/bias*
_output_shapes
:Q*
dtype0
?
(stack0_enc6_middle_contract_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:QQ*9
shared_name*(stack0_enc6_middle_contract_conv0/kernel
?
<stack0_enc6_middle_contract_conv0/kernel/Read/ReadVariableOpReadVariableOp(stack0_enc6_middle_contract_conv0/kernel*&
_output_shapes
:QQ*
dtype0
?
&stack0_enc6_middle_contract_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:Q*7
shared_name(&stack0_enc6_middle_contract_conv0/bias
?
:stack0_enc6_middle_contract_conv0/bias/Read/ReadVariableOpReadVariableOp&stack0_enc6_middle_contract_conv0/bias*
_output_shapes
:Q*
dtype0
?
)stack0_dec0_s16_to_s8_refine_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?6*:
shared_name+)stack0_dec0_s16_to_s8_refine_conv0/kernel
?
=stack0_dec0_s16_to_s8_refine_conv0/kernel/Read/ReadVariableOpReadVariableOp)stack0_dec0_s16_to_s8_refine_conv0/kernel*'
_output_shapes
:?6*
dtype0
?
'stack0_dec0_s16_to_s8_refine_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*8
shared_name)'stack0_dec0_s16_to_s8_refine_conv0/bias
?
;stack0_dec0_s16_to_s8_refine_conv0/bias/Read/ReadVariableOpReadVariableOp'stack0_dec0_s16_to_s8_refine_conv0/bias*
_output_shapes
:6*
dtype0
?
)stack0_dec0_s16_to_s8_refine_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:66*:
shared_name+)stack0_dec0_s16_to_s8_refine_conv1/kernel
?
=stack0_dec0_s16_to_s8_refine_conv1/kernel/Read/ReadVariableOpReadVariableOp)stack0_dec0_s16_to_s8_refine_conv1/kernel*&
_output_shapes
:66*
dtype0
?
'stack0_dec0_s16_to_s8_refine_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:6*8
shared_name)'stack0_dec0_s16_to_s8_refine_conv1/bias
?
;stack0_dec0_s16_to_s8_refine_conv1/bias/Read/ReadVariableOpReadVariableOp'stack0_dec0_s16_to_s8_refine_conv1/bias*
_output_shapes
:6*
dtype0
?
(stack0_dec1_s8_to_s4_refine_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:Z$*9
shared_name*(stack0_dec1_s8_to_s4_refine_conv0/kernel
?
<stack0_dec1_s8_to_s4_refine_conv0/kernel/Read/ReadVariableOpReadVariableOp(stack0_dec1_s8_to_s4_refine_conv0/kernel*&
_output_shapes
:Z$*
dtype0
?
&stack0_dec1_s8_to_s4_refine_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*7
shared_name(&stack0_dec1_s8_to_s4_refine_conv0/bias
?
:stack0_dec1_s8_to_s4_refine_conv0/bias/Read/ReadVariableOpReadVariableOp&stack0_dec1_s8_to_s4_refine_conv0/bias*
_output_shapes
:$*
dtype0
?
(stack0_dec1_s8_to_s4_refine_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:$$*9
shared_name*(stack0_dec1_s8_to_s4_refine_conv1/kernel
?
<stack0_dec1_s8_to_s4_refine_conv1/kernel/Read/ReadVariableOpReadVariableOp(stack0_dec1_s8_to_s4_refine_conv1/kernel*&
_output_shapes
:$$*
dtype0
?
&stack0_dec1_s8_to_s4_refine_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*7
shared_name(&stack0_dec1_s8_to_s4_refine_conv1/bias
?
:stack0_dec1_s8_to_s4_refine_conv1/bias/Read/ReadVariableOpReadVariableOp&stack0_dec1_s8_to_s4_refine_conv1/bias*
_output_shapes
:$*
dtype0
?
(stack0_dec2_s4_to_s2_refine_conv0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*9
shared_name*(stack0_dec2_s4_to_s2_refine_conv0/kernel
?
<stack0_dec2_s4_to_s2_refine_conv0/kernel/Read/ReadVariableOpReadVariableOp(stack0_dec2_s4_to_s2_refine_conv0/kernel*&
_output_shapes
:<*
dtype0
?
&stack0_dec2_s4_to_s2_refine_conv0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&stack0_dec2_s4_to_s2_refine_conv0/bias
?
:stack0_dec2_s4_to_s2_refine_conv0/bias/Read/ReadVariableOpReadVariableOp&stack0_dec2_s4_to_s2_refine_conv0/bias*
_output_shapes
:*
dtype0
?
(stack0_dec2_s4_to_s2_refine_conv1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*9
shared_name*(stack0_dec2_s4_to_s2_refine_conv1/kernel
?
<stack0_dec2_s4_to_s2_refine_conv1/kernel/Read/ReadVariableOpReadVariableOp(stack0_dec2_s4_to_s2_refine_conv1/kernel*&
_output_shapes
:*
dtype0
?
&stack0_dec2_s4_to_s2_refine_conv1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&stack0_dec2_s4_to_s2_refine_conv1/bias
?
:stack0_dec2_s4_to_s2_refine_conv1/bias/Read/ReadVariableOpReadVariableOp&stack0_dec2_s4_to_s2_refine_conv1/bias*
_output_shapes
:*
dtype0
?
CentroidConfmapsHead/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_nameCentroidConfmapsHead/kernel
?
/CentroidConfmapsHead/kernel/Read/ReadVariableOpReadVariableOpCentroidConfmapsHead/kernel*&
_output_shapes
:*
dtype0
?
CentroidConfmapsHead/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameCentroidConfmapsHead/bias
?
-CentroidConfmapsHead/bias/Read/ReadVariableOpReadVariableOpCentroidConfmapsHead/bias*
_output_shapes
:*
dtype0

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*˕
value??B?? B??
?	
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
layer-24
layer-25
layer-26
layer_with_weights-10
layer-27
layer-28
layer_with_weights-11
layer-29
layer-30
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer-34
$layer_with_weights-13
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-14
(layer-39
)layer-40
*layer_with_weights-15
*layer-41
+layer-42
,layer_with_weights-16
,layer-43
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1
signatures
 
h

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
R
8	variables
9trainable_variables
:regularization_losses
;	keras_api
h

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
R
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
R
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
h

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
R
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
h

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
R
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
R
^	variables
_trainable_variables
`regularization_losses
a	keras_api
h

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
R
h	variables
itrainable_variables
jregularization_losses
k	keras_api
h

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
R
r	variables
strainable_variables
tregularization_losses
u	keras_api
R
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
h

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
V
?	variables
?trainable_variables
?regularization_losses
?	keras_api
n
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?
20
31
<2
=3
J4
K5
T6
U7
b8
c9
l10
m11
z12
{13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
?
20
31
<2
=3
J4
K5
T6
U7
b8
c9
l10
m11
z12
{13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33
 
?
-	variables
?layer_metrics
?metrics
.trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layers
/regularization_losses
 
db
VARIABLE_VALUEstack0_enc0_conv0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc0_conv0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

20
31

20
31
 
?
4	variables
?layer_metrics
5trainable_variables
6regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
8	variables
?layer_metrics
9trainable_variables
:regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc0_conv1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc0_conv1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

<0
=1

<0
=1
 
?
>	variables
?layer_metrics
?trainable_variables
@regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
B	variables
?layer_metrics
Ctrainable_variables
Dregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
F	variables
?layer_metrics
Gtrainable_variables
Hregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc1_conv0/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc1_conv0/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

J0
K1

J0
K1
 
?
L	variables
?layer_metrics
Mtrainable_variables
Nregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
P	variables
?layer_metrics
Qtrainable_variables
Rregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc1_conv1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc1_conv1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

T0
U1

T0
U1
 
?
V	variables
?layer_metrics
Wtrainable_variables
Xregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
Z	variables
?layer_metrics
[trainable_variables
\regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
^	variables
?layer_metrics
_trainable_variables
`regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc2_conv0/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc2_conv0/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

b0
c1

b0
c1
 
?
d	variables
?layer_metrics
etrainable_variables
fregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
h	variables
?layer_metrics
itrainable_variables
jregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc2_conv1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc2_conv1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

l0
m1

l0
m1
 
?
n	variables
?layer_metrics
otrainable_variables
pregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
r	variables
?layer_metrics
strainable_variables
tregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
v	variables
?layer_metrics
wtrainable_variables
xregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc3_conv0/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc3_conv0/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

z0
{1

z0
{1
 
?
|	variables
?layer_metrics
}trainable_variables
~regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
db
VARIABLE_VALUEstack0_enc3_conv1/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEstack0_enc3_conv1/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
rp
VARIABLE_VALUE&stack0_enc5_middle_expand_conv0/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE$stack0_enc5_middle_expand_conv0/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
tr
VARIABLE_VALUE(stack0_enc6_middle_contract_conv0/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUE&stack0_enc6_middle_contract_conv0/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
vt
VARIABLE_VALUE)stack0_dec0_s16_to_s8_refine_conv0/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE'stack0_dec0_s16_to_s8_refine_conv0/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
vt
VARIABLE_VALUE)stack0_dec0_s16_to_s8_refine_conv1/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE'stack0_dec0_s16_to_s8_refine_conv1/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
us
VARIABLE_VALUE(stack0_dec1_s8_to_s4_refine_conv0/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE&stack0_dec1_s8_to_s4_refine_conv0/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
us
VARIABLE_VALUE(stack0_dec1_s8_to_s4_refine_conv1/kernel7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE&stack0_dec1_s8_to_s4_refine_conv1/bias5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
us
VARIABLE_VALUE(stack0_dec2_s4_to_s2_refine_conv0/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE&stack0_dec2_s4_to_s2_refine_conv0/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
us
VARIABLE_VALUE(stack0_dec2_s4_to_s2_refine_conv1/kernel7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUE
qo
VARIABLE_VALUE&stack0_dec2_s4_to_s2_refine_conv1/bias5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
hf
VARIABLE_VALUECentroidConfmapsHead/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUECentroidConfmapsHead/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?0
?1
 
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
 
 
 
 
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_inputPlaceholder*1
_output_shapes
:???????????*
dtype0*&
shape:???????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_inputstack0_enc0_conv0/kernelstack0_enc0_conv0/biasstack0_enc0_conv1/kernelstack0_enc0_conv1/biasstack0_enc1_conv0/kernelstack0_enc1_conv0/biasstack0_enc1_conv1/kernelstack0_enc1_conv1/biasstack0_enc2_conv0/kernelstack0_enc2_conv0/biasstack0_enc2_conv1/kernelstack0_enc2_conv1/biasstack0_enc3_conv0/kernelstack0_enc3_conv0/biasstack0_enc3_conv1/kernelstack0_enc3_conv1/bias&stack0_enc5_middle_expand_conv0/kernel$stack0_enc5_middle_expand_conv0/bias(stack0_enc6_middle_contract_conv0/kernel&stack0_enc6_middle_contract_conv0/bias)stack0_dec0_s16_to_s8_refine_conv0/kernel'stack0_dec0_s16_to_s8_refine_conv0/bias)stack0_dec0_s16_to_s8_refine_conv1/kernel'stack0_dec0_s16_to_s8_refine_conv1/bias(stack0_dec1_s8_to_s4_refine_conv0/kernel&stack0_dec1_s8_to_s4_refine_conv0/bias(stack0_dec1_s8_to_s4_refine_conv1/kernel&stack0_dec1_s8_to_s4_refine_conv1/bias(stack0_dec2_s4_to_s2_refine_conv0/kernel&stack0_dec2_s4_to_s2_refine_conv0/bias(stack0_dec2_s4_to_s2_refine_conv1/kernel&stack0_dec2_s4_to_s2_refine_conv1/biasCentroidConfmapsHead/kernelCentroidConfmapsHead/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *,
f'R%
#__inference_signature_wrapper_20101
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename,stack0_enc0_conv0/kernel/Read/ReadVariableOp*stack0_enc0_conv0/bias/Read/ReadVariableOp,stack0_enc0_conv1/kernel/Read/ReadVariableOp*stack0_enc0_conv1/bias/Read/ReadVariableOp,stack0_enc1_conv0/kernel/Read/ReadVariableOp*stack0_enc1_conv0/bias/Read/ReadVariableOp,stack0_enc1_conv1/kernel/Read/ReadVariableOp*stack0_enc1_conv1/bias/Read/ReadVariableOp,stack0_enc2_conv0/kernel/Read/ReadVariableOp*stack0_enc2_conv0/bias/Read/ReadVariableOp,stack0_enc2_conv1/kernel/Read/ReadVariableOp*stack0_enc2_conv1/bias/Read/ReadVariableOp,stack0_enc3_conv0/kernel/Read/ReadVariableOp*stack0_enc3_conv0/bias/Read/ReadVariableOp,stack0_enc3_conv1/kernel/Read/ReadVariableOp*stack0_enc3_conv1/bias/Read/ReadVariableOp:stack0_enc5_middle_expand_conv0/kernel/Read/ReadVariableOp8stack0_enc5_middle_expand_conv0/bias/Read/ReadVariableOp<stack0_enc6_middle_contract_conv0/kernel/Read/ReadVariableOp:stack0_enc6_middle_contract_conv0/bias/Read/ReadVariableOp=stack0_dec0_s16_to_s8_refine_conv0/kernel/Read/ReadVariableOp;stack0_dec0_s16_to_s8_refine_conv0/bias/Read/ReadVariableOp=stack0_dec0_s16_to_s8_refine_conv1/kernel/Read/ReadVariableOp;stack0_dec0_s16_to_s8_refine_conv1/bias/Read/ReadVariableOp<stack0_dec1_s8_to_s4_refine_conv0/kernel/Read/ReadVariableOp:stack0_dec1_s8_to_s4_refine_conv0/bias/Read/ReadVariableOp<stack0_dec1_s8_to_s4_refine_conv1/kernel/Read/ReadVariableOp:stack0_dec1_s8_to_s4_refine_conv1/bias/Read/ReadVariableOp<stack0_dec2_s4_to_s2_refine_conv0/kernel/Read/ReadVariableOp:stack0_dec2_s4_to_s2_refine_conv0/bias/Read/ReadVariableOp<stack0_dec2_s4_to_s2_refine_conv1/kernel/Read/ReadVariableOp:stack0_dec2_s4_to_s2_refine_conv1/bias/Read/ReadVariableOp/CentroidConfmapsHead/kernel/Read/ReadVariableOp-CentroidConfmapsHead/bias/Read/ReadVariableOpConst*/
Tin(
&2$*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *'
f"R 
__inference__traced_save_21182
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamestack0_enc0_conv0/kernelstack0_enc0_conv0/biasstack0_enc0_conv1/kernelstack0_enc0_conv1/biasstack0_enc1_conv0/kernelstack0_enc1_conv0/biasstack0_enc1_conv1/kernelstack0_enc1_conv1/biasstack0_enc2_conv0/kernelstack0_enc2_conv0/biasstack0_enc2_conv1/kernelstack0_enc2_conv1/biasstack0_enc3_conv0/kernelstack0_enc3_conv0/biasstack0_enc3_conv1/kernelstack0_enc3_conv1/bias&stack0_enc5_middle_expand_conv0/kernel$stack0_enc5_middle_expand_conv0/bias(stack0_enc6_middle_contract_conv0/kernel&stack0_enc6_middle_contract_conv0/bias)stack0_dec0_s16_to_s8_refine_conv0/kernel'stack0_dec0_s16_to_s8_refine_conv0/bias)stack0_dec0_s16_to_s8_refine_conv1/kernel'stack0_dec0_s16_to_s8_refine_conv1/bias(stack0_dec1_s8_to_s4_refine_conv0/kernel&stack0_dec1_s8_to_s4_refine_conv0/bias(stack0_dec1_s8_to_s4_refine_conv1/kernel&stack0_dec1_s8_to_s4_refine_conv1/bias(stack0_dec2_s4_to_s2_refine_conv0/kernel&stack0_dec2_s4_to_s2_refine_conv0/bias(stack0_dec2_s4_to_s2_refine_conv1/kernel&stack0_dec2_s4_to_s2_refine_conv1/biasCentroidConfmapsHead/kernelCentroidConfmapsHead/bias*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__traced_restore_21294??
?P
?
__inference__traced_save_21182
file_prefix7
3savev2_stack0_enc0_conv0_kernel_read_readvariableop5
1savev2_stack0_enc0_conv0_bias_read_readvariableop7
3savev2_stack0_enc0_conv1_kernel_read_readvariableop5
1savev2_stack0_enc0_conv1_bias_read_readvariableop7
3savev2_stack0_enc1_conv0_kernel_read_readvariableop5
1savev2_stack0_enc1_conv0_bias_read_readvariableop7
3savev2_stack0_enc1_conv1_kernel_read_readvariableop5
1savev2_stack0_enc1_conv1_bias_read_readvariableop7
3savev2_stack0_enc2_conv0_kernel_read_readvariableop5
1savev2_stack0_enc2_conv0_bias_read_readvariableop7
3savev2_stack0_enc2_conv1_kernel_read_readvariableop5
1savev2_stack0_enc2_conv1_bias_read_readvariableop7
3savev2_stack0_enc3_conv0_kernel_read_readvariableop5
1savev2_stack0_enc3_conv0_bias_read_readvariableop7
3savev2_stack0_enc3_conv1_kernel_read_readvariableop5
1savev2_stack0_enc3_conv1_bias_read_readvariableopE
Asavev2_stack0_enc5_middle_expand_conv0_kernel_read_readvariableopC
?savev2_stack0_enc5_middle_expand_conv0_bias_read_readvariableopG
Csavev2_stack0_enc6_middle_contract_conv0_kernel_read_readvariableopE
Asavev2_stack0_enc6_middle_contract_conv0_bias_read_readvariableopH
Dsavev2_stack0_dec0_s16_to_s8_refine_conv0_kernel_read_readvariableopF
Bsavev2_stack0_dec0_s16_to_s8_refine_conv0_bias_read_readvariableopH
Dsavev2_stack0_dec0_s16_to_s8_refine_conv1_kernel_read_readvariableopF
Bsavev2_stack0_dec0_s16_to_s8_refine_conv1_bias_read_readvariableopG
Csavev2_stack0_dec1_s8_to_s4_refine_conv0_kernel_read_readvariableopE
Asavev2_stack0_dec1_s8_to_s4_refine_conv0_bias_read_readvariableopG
Csavev2_stack0_dec1_s8_to_s4_refine_conv1_kernel_read_readvariableopE
Asavev2_stack0_dec1_s8_to_s4_refine_conv1_bias_read_readvariableopG
Csavev2_stack0_dec2_s4_to_s2_refine_conv0_kernel_read_readvariableopE
Asavev2_stack0_dec2_s4_to_s2_refine_conv0_bias_read_readvariableopG
Csavev2_stack0_dec2_s4_to_s2_refine_conv1_kernel_read_readvariableopE
Asavev2_stack0_dec2_s4_to_s2_refine_conv1_bias_read_readvariableop:
6savev2_centroidconfmapshead_kernel_read_readvariableop8
4savev2_centroidconfmapshead_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:03savev2_stack0_enc0_conv0_kernel_read_readvariableop1savev2_stack0_enc0_conv0_bias_read_readvariableop3savev2_stack0_enc0_conv1_kernel_read_readvariableop1savev2_stack0_enc0_conv1_bias_read_readvariableop3savev2_stack0_enc1_conv0_kernel_read_readvariableop1savev2_stack0_enc1_conv0_bias_read_readvariableop3savev2_stack0_enc1_conv1_kernel_read_readvariableop1savev2_stack0_enc1_conv1_bias_read_readvariableop3savev2_stack0_enc2_conv0_kernel_read_readvariableop1savev2_stack0_enc2_conv0_bias_read_readvariableop3savev2_stack0_enc2_conv1_kernel_read_readvariableop1savev2_stack0_enc2_conv1_bias_read_readvariableop3savev2_stack0_enc3_conv0_kernel_read_readvariableop1savev2_stack0_enc3_conv0_bias_read_readvariableop3savev2_stack0_enc3_conv1_kernel_read_readvariableop1savev2_stack0_enc3_conv1_bias_read_readvariableopAsavev2_stack0_enc5_middle_expand_conv0_kernel_read_readvariableop?savev2_stack0_enc5_middle_expand_conv0_bias_read_readvariableopCsavev2_stack0_enc6_middle_contract_conv0_kernel_read_readvariableopAsavev2_stack0_enc6_middle_contract_conv0_bias_read_readvariableopDsavev2_stack0_dec0_s16_to_s8_refine_conv0_kernel_read_readvariableopBsavev2_stack0_dec0_s16_to_s8_refine_conv0_bias_read_readvariableopDsavev2_stack0_dec0_s16_to_s8_refine_conv1_kernel_read_readvariableopBsavev2_stack0_dec0_s16_to_s8_refine_conv1_bias_read_readvariableopCsavev2_stack0_dec1_s8_to_s4_refine_conv0_kernel_read_readvariableopAsavev2_stack0_dec1_s8_to_s4_refine_conv0_bias_read_readvariableopCsavev2_stack0_dec1_s8_to_s4_refine_conv1_kernel_read_readvariableopAsavev2_stack0_dec1_s8_to_s4_refine_conv1_bias_read_readvariableopCsavev2_stack0_dec2_s4_to_s2_refine_conv0_kernel_read_readvariableopAsavev2_stack0_dec2_s4_to_s2_refine_conv0_bias_read_readvariableopCsavev2_stack0_dec2_s4_to_s2_refine_conv1_kernel_read_readvariableopAsavev2_stack0_dec2_s4_to_s2_refine_conv1_bias_read_readvariableop6savev2_centroidconfmapshead_kernel_read_readvariableop4savev2_centroidconfmapshead_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *1
dtypes'
%2#2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :::::::::$:$:$$:$:$6:6:66:6:6Q:Q:QQ:Q:?6:6:66:6:Z$:$:$$:$:<:::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
:: 

_output_shapes
::,	(
&
_output_shapes
:$: 


_output_shapes
:$:,(
&
_output_shapes
:$$: 

_output_shapes
:$:,(
&
_output_shapes
:$6: 

_output_shapes
:6:,(
&
_output_shapes
:66: 

_output_shapes
:6:,(
&
_output_shapes
:6Q: 

_output_shapes
:Q:,(
&
_output_shapes
:QQ: 

_output_shapes
:Q:-)
'
_output_shapes
:?6: 

_output_shapes
:6:,(
&
_output_shapes
:66: 

_output_shapes
:6:,(
&
_output_shapes
:Z$: 

_output_shapes
:$:,(
&
_output_shapes
:$$: 

_output_shapes
:$:,(
&
_output_shapes
:<: 

_output_shapes
::,(
&
_output_shapes
::  

_output_shapes
::,!(
&
_output_shapes
:: "

_output_shapes
::#

_output_shapes
: 
?
?
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_20838
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????@@6:+???????????????????????????Q:Y U
/
_output_shapes
:?????????@@6
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????Q
"
_user_specified_name
inputs/1
?
?	
,__inference_functional_1_layer_call_fn_20174

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:$
	unknown_8:$#
	unknown_9:$$

unknown_10:$$

unknown_11:$6

unknown_12:6$

unknown_13:66

unknown_14:6$

unknown_15:6Q

unknown_16:Q$

unknown_17:QQ

unknown_18:Q%

unknown_19:?6

unknown_20:6$

unknown_21:66

unknown_22:6$

unknown_23:Z$

unknown_24:$$

unknown_25:$$

unknown_26:$$

unknown_27:<

unknown_28:$

unknown_29:

unknown_30:$

unknown_31:

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_191042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_18800

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_18729

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_18812

inputs8
conv2d_readvariableop_resource:$$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc2_conv1_layer_call_fn_20689

inputs!
unknown:$$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_188122
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_18836

inputs8
conv2d_readvariableop_resource:$6-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$6*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@$
 
_user_specified_nameinputs
?

?
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_20857

inputs9
conv2d_readvariableop_resource:?6-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?6*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
?
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_20980
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????<2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:+???????????????????????????$:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????$
"
_user_specified_name
inputs/1
?

?
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_18995

inputs8
conv2d_readvariableop_resource:Z$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:Z$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????Z
 
_user_specified_nameinputs
?

?
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_18906

inputs8
conv2d_readvariableop_resource:QQ-
biasadd_readvariableop_resource:Q
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:QQ*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  Q: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
?
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20896

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
|
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_18634

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
A__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_fn_20918

inputs!
unknown:Z$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_189952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????Z: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????Z
 
_user_specified_nameinputs
??
?
!__inference__traced_restore_21294
file_prefixC
)assignvariableop_stack0_enc0_conv0_kernel:7
)assignvariableop_1_stack0_enc0_conv0_bias:E
+assignvariableop_2_stack0_enc0_conv1_kernel:7
)assignvariableop_3_stack0_enc0_conv1_bias:E
+assignvariableop_4_stack0_enc1_conv0_kernel:7
)assignvariableop_5_stack0_enc1_conv0_bias:E
+assignvariableop_6_stack0_enc1_conv1_kernel:7
)assignvariableop_7_stack0_enc1_conv1_bias:E
+assignvariableop_8_stack0_enc2_conv0_kernel:$7
)assignvariableop_9_stack0_enc2_conv0_bias:$F
,assignvariableop_10_stack0_enc2_conv1_kernel:$$8
*assignvariableop_11_stack0_enc2_conv1_bias:$F
,assignvariableop_12_stack0_enc3_conv0_kernel:$68
*assignvariableop_13_stack0_enc3_conv0_bias:6F
,assignvariableop_14_stack0_enc3_conv1_kernel:668
*assignvariableop_15_stack0_enc3_conv1_bias:6T
:assignvariableop_16_stack0_enc5_middle_expand_conv0_kernel:6QF
8assignvariableop_17_stack0_enc5_middle_expand_conv0_bias:QV
<assignvariableop_18_stack0_enc6_middle_contract_conv0_kernel:QQH
:assignvariableop_19_stack0_enc6_middle_contract_conv0_bias:QX
=assignvariableop_20_stack0_dec0_s16_to_s8_refine_conv0_kernel:?6I
;assignvariableop_21_stack0_dec0_s16_to_s8_refine_conv0_bias:6W
=assignvariableop_22_stack0_dec0_s16_to_s8_refine_conv1_kernel:66I
;assignvariableop_23_stack0_dec0_s16_to_s8_refine_conv1_bias:6V
<assignvariableop_24_stack0_dec1_s8_to_s4_refine_conv0_kernel:Z$H
:assignvariableop_25_stack0_dec1_s8_to_s4_refine_conv0_bias:$V
<assignvariableop_26_stack0_dec1_s8_to_s4_refine_conv1_kernel:$$H
:assignvariableop_27_stack0_dec1_s8_to_s4_refine_conv1_bias:$V
<assignvariableop_28_stack0_dec2_s4_to_s2_refine_conv0_kernel:<H
:assignvariableop_29_stack0_dec2_s4_to_s2_refine_conv0_bias:V
<assignvariableop_30_stack0_dec2_s4_to_s2_refine_conv1_kernel:H
:assignvariableop_31_stack0_dec2_s4_to_s2_refine_conv1_bias:I
/assignvariableop_32_centroidconfmapshead_kernel:;
-assignvariableop_33_centroidconfmapshead_bias:
identity_35??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*?
value?B?#B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-13/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-15/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:#*
dtype0*Y
valuePBN#B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::*1
dtypes'
%2#2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp)assignvariableop_stack0_enc0_conv0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp)assignvariableop_1_stack0_enc0_conv0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp+assignvariableop_2_stack0_enc0_conv1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp)assignvariableop_3_stack0_enc0_conv1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp+assignvariableop_4_stack0_enc1_conv0_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp)assignvariableop_5_stack0_enc1_conv0_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp+assignvariableop_6_stack0_enc1_conv1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp)assignvariableop_7_stack0_enc1_conv1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp+assignvariableop_8_stack0_enc2_conv0_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp)assignvariableop_9_stack0_enc2_conv0_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp,assignvariableop_10_stack0_enc2_conv1_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp*assignvariableop_11_stack0_enc2_conv1_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp,assignvariableop_12_stack0_enc3_conv0_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_stack0_enc3_conv0_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp,assignvariableop_14_stack0_enc3_conv1_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp*assignvariableop_15_stack0_enc3_conv1_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp:assignvariableop_16_stack0_enc5_middle_expand_conv0_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp8assignvariableop_17_stack0_enc5_middle_expand_conv0_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp<assignvariableop_18_stack0_enc6_middle_contract_conv0_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp:assignvariableop_19_stack0_enc6_middle_contract_conv0_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOp=assignvariableop_20_stack0_dec0_s16_to_s8_refine_conv0_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp;assignvariableop_21_stack0_dec0_s16_to_s8_refine_conv0_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp=assignvariableop_22_stack0_dec0_s16_to_s8_refine_conv1_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp;assignvariableop_23_stack0_dec0_s16_to_s8_refine_conv1_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp<assignvariableop_24_stack0_dec1_s8_to_s4_refine_conv0_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp:assignvariableop_25_stack0_dec1_s8_to_s4_refine_conv0_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp<assignvariableop_26_stack0_dec1_s8_to_s4_refine_conv1_kernelIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp:assignvariableop_27_stack0_dec1_s8_to_s4_refine_conv1_biasIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp<assignvariableop_28_stack0_dec2_s4_to_s2_refine_conv0_kernelIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp:assignvariableop_29_stack0_dec2_s4_to_s2_refine_conv0_biasIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp<assignvariableop_30_stack0_dec2_s4_to_s2_refine_conv1_kernelIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp:assignvariableop_31_stack0_dec2_s4_to_s2_refine_conv1_biasIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp/assignvariableop_32_centroidconfmapshead_kernelIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_centroidconfmapshead_biasIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_339
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_34Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_34?
Identity_35IdentityIdentity_34:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_35"#
identity_35Identity_35:output:0*Y
_input_shapesH
F: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?

?
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_21057

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?	
,__inference_functional_1_layer_call_fn_20247

inputs!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:$
	unknown_8:$#
	unknown_9:$$

unknown_10:$$

unknown_11:$6

unknown_12:6$

unknown_13:66

unknown_14:6$

unknown_15:6Q

unknown_16:Q$

unknown_17:QQ

unknown_18:Q%

unknown_19:?6

unknown_20:6$

unknown_21:66

unknown_22:6$

unknown_23:Z$

unknown_24:$$

unknown_25:$$

unknown_26:$$

unknown_27:<

unknown_28:$

unknown_29:

unknown_30:$

unknown_31:

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_196522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_20612

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc2_act0_relu_layer_call_fn_20675

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_188002
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_20564

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_18765

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_19085

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_stack0_enc6_middle_contract_conv0_layer_call_fn_20805

inputs!
unknown:QQ
	unknown_0:Q
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_189062
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  Q: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_20738

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
?
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_18973

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
`
D__inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_fn_18678

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_186722
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_20026	
input1
stack0_enc0_conv0_19914:%
stack0_enc0_conv0_19916:1
stack0_enc0_conv1_19920:%
stack0_enc0_conv1_19922:1
stack0_enc1_conv0_19927:%
stack0_enc1_conv0_19929:1
stack0_enc1_conv1_19933:%
stack0_enc1_conv1_19935:1
stack0_enc2_conv0_19940:$%
stack0_enc2_conv0_19942:$1
stack0_enc2_conv1_19946:$$%
stack0_enc2_conv1_19948:$1
stack0_enc3_conv0_19953:$6%
stack0_enc3_conv0_19955:61
stack0_enc3_conv1_19959:66%
stack0_enc3_conv1_19961:6?
%stack0_enc5_middle_expand_conv0_19966:6Q3
%stack0_enc5_middle_expand_conv0_19968:QA
'stack0_enc6_middle_contract_conv0_19972:QQ5
'stack0_enc6_middle_contract_conv0_19974:QC
(stack0_dec0_s16_to_s8_refine_conv0_19980:?66
(stack0_dec0_s16_to_s8_refine_conv0_19982:6B
(stack0_dec0_s16_to_s8_refine_conv1_19986:666
(stack0_dec0_s16_to_s8_refine_conv1_19988:6A
'stack0_dec1_s8_to_s4_refine_conv0_19994:Z$5
'stack0_dec1_s8_to_s4_refine_conv0_19996:$A
'stack0_dec1_s8_to_s4_refine_conv1_20000:$$5
'stack0_dec1_s8_to_s4_refine_conv1_20002:$A
'stack0_dec2_s4_to_s2_refine_conv0_20008:<5
'stack0_dec2_s4_to_s2_refine_conv0_20010:A
'stack0_dec2_s4_to_s2_refine_conv1_20014:5
'stack0_dec2_s4_to_s2_refine_conv1_20016:4
centroidconfmapshead_20020:(
centroidconfmapshead_20022:
identity??,CentroidConfmapsHead/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?)stack0_enc0_conv0/StatefulPartitionedCall?)stack0_enc0_conv1/StatefulPartitionedCall?)stack0_enc1_conv0/StatefulPartitionedCall?)stack0_enc1_conv1/StatefulPartitionedCall?)stack0_enc2_conv0/StatefulPartitionedCall?)stack0_enc2_conv1/StatefulPartitionedCall?)stack0_enc3_conv0/StatefulPartitionedCall?)stack0_enc3_conv1/StatefulPartitionedCall?7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
)stack0_enc0_conv0/StatefulPartitionedCallStatefulPartitionedCallinputstack0_enc0_conv0_19914stack0_enc0_conv0_19916*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_186952+
)stack0_enc0_conv0/StatefulPartitionedCall?
%stack0_enc0_act0_relu/PartitionedCallPartitionedCall2stack0_enc0_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_187062'
%stack0_enc0_act0_relu/PartitionedCall?
)stack0_enc0_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc0_act0_relu/PartitionedCall:output:0stack0_enc0_conv1_19920stack0_enc0_conv1_19922*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_187182+
)stack0_enc0_conv1/StatefulPartitionedCall?
%stack0_enc0_act1_relu/PartitionedCallPartitionedCall2stack0_enc0_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_187292'
%stack0_enc0_act1_relu/PartitionedCall?
 stack0_enc1_pool/PartitionedCallPartitionedCall.stack0_enc0_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_185792"
 stack0_enc1_pool/PartitionedCall?
)stack0_enc1_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc1_pool/PartitionedCall:output:0stack0_enc1_conv0_19927stack0_enc1_conv0_19929*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_187422+
)stack0_enc1_conv0/StatefulPartitionedCall?
%stack0_enc1_act0_relu/PartitionedCallPartitionedCall2stack0_enc1_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_187532'
%stack0_enc1_act0_relu/PartitionedCall?
)stack0_enc1_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc1_act0_relu/PartitionedCall:output:0stack0_enc1_conv1_19933stack0_enc1_conv1_19935*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_187652+
)stack0_enc1_conv1/StatefulPartitionedCall?
%stack0_enc1_act1_relu/PartitionedCallPartitionedCall2stack0_enc1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_187762'
%stack0_enc1_act1_relu/PartitionedCall?
 stack0_enc2_pool/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_185912"
 stack0_enc2_pool/PartitionedCall?
)stack0_enc2_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc2_pool/PartitionedCall:output:0stack0_enc2_conv0_19940stack0_enc2_conv0_19942*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_187892+
)stack0_enc2_conv0/StatefulPartitionedCall?
%stack0_enc2_act0_relu/PartitionedCallPartitionedCall2stack0_enc2_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_188002'
%stack0_enc2_act0_relu/PartitionedCall?
)stack0_enc2_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc2_act0_relu/PartitionedCall:output:0stack0_enc2_conv1_19946stack0_enc2_conv1_19948*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_188122+
)stack0_enc2_conv1/StatefulPartitionedCall?
%stack0_enc2_act1_relu/PartitionedCallPartitionedCall2stack0_enc2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_188232'
%stack0_enc2_act1_relu/PartitionedCall?
 stack0_enc3_pool/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_186032"
 stack0_enc3_pool/PartitionedCall?
)stack0_enc3_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc3_pool/PartitionedCall:output:0stack0_enc3_conv0_19953stack0_enc3_conv0_19955*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_188362+
)stack0_enc3_conv0/StatefulPartitionedCall?
%stack0_enc3_act0_relu/PartitionedCallPartitionedCall2stack0_enc3_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_188472'
%stack0_enc3_act0_relu/PartitionedCall?
)stack0_enc3_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc3_act0_relu/PartitionedCall:output:0stack0_enc3_conv1_19959stack0_enc3_conv1_19961*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_188592+
)stack0_enc3_conv1/StatefulPartitionedCall?
%stack0_enc3_act1_relu/PartitionedCallPartitionedCall2stack0_enc3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_188702'
%stack0_enc3_act1_relu/PartitionedCall?
%stack0_enc4_last_pool/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_186152'
%stack0_enc4_last_pool/PartitionedCall?
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc4_last_pool/PartitionedCall:output:0%stack0_enc5_middle_expand_conv0_19966%stack0_enc5_middle_expand_conv0_19968*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_1888329
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?
3stack0_enc5_middle_expand_act0_relu/PartitionedCallPartitionedCall@stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *g
fbR`
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_1889425
3stack0_enc5_middle_expand_act0_relu/PartitionedCall?
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCallStatefulPartitionedCall<stack0_enc5_middle_expand_act0_relu/PartitionedCall:output:0'stack0_enc6_middle_contract_conv0_19972'stack0_enc6_middle_contract_conv0_19974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_189062;
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
5stack0_enc6_middle_contract_act0_relu/PartitionedCallPartitionedCallBstack0_enc6_middle_contract_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_1891727
5stack0_enc6_middle_contract_act0_relu/PartitionedCall?
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCallPartitionedCall>stack0_enc6_middle_contract_act0_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_1863427
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall?
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0>stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_1892723
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall:stack0_dec0_s16_to_s8_skip_concat/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv0_19980(stack0_dec0_s16_to_s8_refine_conv0_19982*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_189392<
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_189502=
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallDstack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv1_19986(stack0_dec0_s16_to_s8_refine_conv1_19988*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_189622<
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_189732=
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall?
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCallPartitionedCallDstack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_1865326
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall?
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0=stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_1898322
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec1_s8_to_s4_skip_concat/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv0_19994'stack0_dec1_s8_to_s4_refine_conv0_19996*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_189952;
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190062<
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv1_20000'stack0_dec1_s8_to_s4_refine_conv1_20002*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_190182;
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190292<
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall?
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCallPartitionedCallCstack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_1867226
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall?
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0=stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_1903922
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec2_s4_to_s2_skip_concat/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv0_20008'stack0_dec2_s4_to_s2_refine_conv0_20010*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_190512;
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190622<
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv1_20014'stack0_dec2_s4_to_s2_refine_conv1_20016*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_190742;
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190852<
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall?
,CentroidConfmapsHead/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall:output:0centroidconfmapshead_20020centroidconfmapshead_20022*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_190972.
,CentroidConfmapsHead/StatefulPartitionedCall?
IdentityIdentity5CentroidConfmapsHead/StatefulPartitionedCall:output:0-^CentroidConfmapsHead/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall*^stack0_enc0_conv0/StatefulPartitionedCall*^stack0_enc0_conv1/StatefulPartitionedCall*^stack0_enc1_conv0/StatefulPartitionedCall*^stack0_enc1_conv1/StatefulPartitionedCall*^stack0_enc2_conv0/StatefulPartitionedCall*^stack0_enc2_conv1/StatefulPartitionedCall*^stack0_enc3_conv0/StatefulPartitionedCall*^stack0_enc3_conv1/StatefulPartitionedCall8^stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:^stack0_enc6_middle_contract_conv0/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,CentroidConfmapsHead/StatefulPartitionedCall,CentroidConfmapsHead/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall2V
)stack0_enc0_conv0/StatefulPartitionedCall)stack0_enc0_conv0/StatefulPartitionedCall2V
)stack0_enc0_conv1/StatefulPartitionedCall)stack0_enc0_conv1/StatefulPartitionedCall2V
)stack0_enc1_conv0/StatefulPartitionedCall)stack0_enc1_conv0/StatefulPartitionedCall2V
)stack0_enc1_conv1/StatefulPartitionedCall)stack0_enc1_conv1/StatefulPartitionedCall2V
)stack0_enc2_conv0/StatefulPartitionedCall)stack0_enc2_conv0/StatefulPartitionedCall2V
)stack0_enc2_conv1/StatefulPartitionedCall)stack0_enc2_conv1/StatefulPartitionedCall2V
)stack0_enc3_conv0/StatefulPartitionedCall)stack0_enc3_conv0/StatefulPartitionedCall2V
)stack0_enc3_conv1/StatefulPartitionedCall)stack0_enc3_conv1/StatefulPartitionedCall2r
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall2v
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
l
@__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_fn_20902
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_189832
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????$:+???????????????????????????6:[ W
1
_output_shapes
:???????????$
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????6
"
_user_specified_name
inputs/1
?
m
A__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_fn_20831
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_189272
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????@@6:+???????????????????????????Q:Y U
/
_output_shapes
:?????????@@6
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????Q
"
_user_specified_name
inputs/1
?
l
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_20767

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_18706

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_18591

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_19039

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????<2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:+???????????????????????????$:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????$
 
_user_specified_nameinputs
?
g
K__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_fn_20862

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_189502
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_20709

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc0_act0_relu_layer_call_fn_20559

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_187062
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_20554

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_stack0_enc2_pool_layer_call_fn_18597

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_185912
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
{
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_18653

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc1_act0_relu_layer_call_fn_20617

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_187532
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_18776

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc3_act1_relu_layer_call_fn_20762

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_188702
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_18742

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_20757

inputs8
conv2d_readvariableop_resource:66-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
|
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_20825

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  Q2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_18718

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc2_conv0_layer_call_fn_20660

inputs!
unknown:$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_187892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_fn_21033

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190852
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_fn_21018

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_190742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc0_act1_relu_layer_call_fn_20588

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_187292
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_21038

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_20786

inputs8
conv2d_readvariableop_resource:6Q-
biasadd_readvariableop_resource:Q
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:6Q*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  6
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_19074

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_20886

inputs8
conv2d_readvariableop_resource:66-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
g
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_18579

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
L
0__inference_stack0_enc1_pool_layer_call_fn_18585

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_185792
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?)
 __inference__wrapped_model_18573	
inputW
=functional_1_stack0_enc0_conv0_conv2d_readvariableop_resource:L
>functional_1_stack0_enc0_conv0_biasadd_readvariableop_resource:W
=functional_1_stack0_enc0_conv1_conv2d_readvariableop_resource:L
>functional_1_stack0_enc0_conv1_biasadd_readvariableop_resource:W
=functional_1_stack0_enc1_conv0_conv2d_readvariableop_resource:L
>functional_1_stack0_enc1_conv0_biasadd_readvariableop_resource:W
=functional_1_stack0_enc1_conv1_conv2d_readvariableop_resource:L
>functional_1_stack0_enc1_conv1_biasadd_readvariableop_resource:W
=functional_1_stack0_enc2_conv0_conv2d_readvariableop_resource:$L
>functional_1_stack0_enc2_conv0_biasadd_readvariableop_resource:$W
=functional_1_stack0_enc2_conv1_conv2d_readvariableop_resource:$$L
>functional_1_stack0_enc2_conv1_biasadd_readvariableop_resource:$W
=functional_1_stack0_enc3_conv0_conv2d_readvariableop_resource:$6L
>functional_1_stack0_enc3_conv0_biasadd_readvariableop_resource:6W
=functional_1_stack0_enc3_conv1_conv2d_readvariableop_resource:66L
>functional_1_stack0_enc3_conv1_biasadd_readvariableop_resource:6e
Kfunctional_1_stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource:6QZ
Lfunctional_1_stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource:Qg
Mfunctional_1_stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource:QQ\
Nfunctional_1_stack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource:Qi
Nfunctional_1_stack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource:?6]
Ofunctional_1_stack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource:6h
Nfunctional_1_stack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource:66]
Ofunctional_1_stack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource:6g
Mfunctional_1_stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource:Z$\
Nfunctional_1_stack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource:$g
Mfunctional_1_stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource:$$\
Nfunctional_1_stack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource:$g
Mfunctional_1_stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource:<\
Nfunctional_1_stack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource:g
Mfunctional_1_stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource:\
Nfunctional_1_stack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource:Z
@functional_1_centroidconfmapshead_conv2d_readvariableop_resource:O
Afunctional_1_centroidconfmapshead_biasadd_readvariableop_resource:
identity??8functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp?7functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOp?Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?Efunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?Efunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?Efunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?Efunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?Efunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?Efunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?5functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp?4functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp?5functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp?4functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp?5functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp?4functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp?5functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp?4functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp?5functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp?4functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp?5functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp?4functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp?5functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp?4functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp?5functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp?4functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOp?Cfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?Bfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?Efunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?Dfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
4functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc0_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp?
%functional_1/stack0_enc0_conv0/Conv2DConv2Dinput<functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%functional_1/stack0_enc0_conv0/Conv2D?
5functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc0_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc0_conv0/BiasAddBiasAdd.functional_1/stack0_enc0_conv0/Conv2D:output:0=functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&functional_1/stack0_enc0_conv0/BiasAdd?
'functional_1/stack0_enc0_act0_relu/ReluRelu/functional_1/stack0_enc0_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'functional_1/stack0_enc0_act0_relu/Relu?
4functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp?
%functional_1/stack0_enc0_conv1/Conv2DConv2D5functional_1/stack0_enc0_act0_relu/Relu:activations:0<functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%functional_1/stack0_enc0_conv1/Conv2D?
5functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc0_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc0_conv1/BiasAddBiasAdd.functional_1/stack0_enc0_conv1/Conv2D:output:0=functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&functional_1/stack0_enc0_conv1/BiasAdd?
'functional_1/stack0_enc0_act1_relu/ReluRelu/functional_1/stack0_enc0_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'functional_1/stack0_enc0_act1_relu/Relu?
%functional_1/stack0_enc1_pool/MaxPoolMaxPool5functional_1/stack0_enc0_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2'
%functional_1/stack0_enc1_pool/MaxPool?
4functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc1_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp?
%functional_1/stack0_enc1_conv0/Conv2DConv2D.functional_1/stack0_enc1_pool/MaxPool:output:0<functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%functional_1/stack0_enc1_conv0/Conv2D?
5functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc1_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc1_conv0/BiasAddBiasAdd.functional_1/stack0_enc1_conv0/Conv2D:output:0=functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&functional_1/stack0_enc1_conv0/BiasAdd?
'functional_1/stack0_enc1_act0_relu/ReluRelu/functional_1/stack0_enc1_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'functional_1/stack0_enc1_act0_relu/Relu?
4functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype026
4functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp?
%functional_1/stack0_enc1_conv1/Conv2DConv2D5functional_1/stack0_enc1_act0_relu/Relu:activations:0<functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2'
%functional_1/stack0_enc1_conv1/Conv2D?
5functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc1_conv1/BiasAddBiasAdd.functional_1/stack0_enc1_conv1/Conv2D:output:0=functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2(
&functional_1/stack0_enc1_conv1/BiasAdd?
'functional_1/stack0_enc1_act1_relu/ReluRelu/functional_1/stack0_enc1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2)
'functional_1/stack0_enc1_act1_relu/Relu?
%functional_1/stack0_enc2_pool/MaxPoolMaxPool5functional_1/stack0_enc1_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2'
%functional_1/stack0_enc2_pool/MaxPool?
4functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc2_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype026
4functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp?
%functional_1/stack0_enc2_conv0/Conv2DConv2D.functional_1/stack0_enc2_pool/MaxPool:output:0<functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2'
%functional_1/stack0_enc2_conv0/Conv2D?
5functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc2_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype027
5functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc2_conv0/BiasAddBiasAdd.functional_1/stack0_enc2_conv0/Conv2D:output:0=functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2(
&functional_1/stack0_enc2_conv0/BiasAdd?
'functional_1/stack0_enc2_act0_relu/ReluRelu/functional_1/stack0_enc2_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2)
'functional_1/stack0_enc2_act0_relu/Relu?
4functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype026
4functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp?
%functional_1/stack0_enc2_conv1/Conv2DConv2D5functional_1/stack0_enc2_act0_relu/Relu:activations:0<functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2'
%functional_1/stack0_enc2_conv1/Conv2D?
5functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc2_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype027
5functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc2_conv1/BiasAddBiasAdd.functional_1/stack0_enc2_conv1/Conv2D:output:0=functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2(
&functional_1/stack0_enc2_conv1/BiasAdd?
'functional_1/stack0_enc2_act1_relu/ReluRelu/functional_1/stack0_enc2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2)
'functional_1/stack0_enc2_act1_relu/Relu?
%functional_1/stack0_enc3_pool/MaxPoolMaxPool5functional_1/stack0_enc2_act1_relu/Relu:activations:0*/
_output_shapes
:?????????@@$*
ksize
*
paddingSAME*
strides
2'
%functional_1/stack0_enc3_pool/MaxPool?
4functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc3_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$6*
dtype026
4functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp?
%functional_1/stack0_enc3_conv0/Conv2DConv2D.functional_1/stack0_enc3_pool/MaxPool:output:0<functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2'
%functional_1/stack0_enc3_conv0/Conv2D?
5functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc3_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype027
5functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc3_conv0/BiasAddBiasAdd.functional_1/stack0_enc3_conv0/Conv2D:output:0=functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62(
&functional_1/stack0_enc3_conv0/BiasAdd?
'functional_1/stack0_enc3_act0_relu/ReluRelu/functional_1/stack0_enc3_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62)
'functional_1/stack0_enc3_act0_relu/Relu?
4functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOpReadVariableOp=functional_1_stack0_enc3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype026
4functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOp?
%functional_1/stack0_enc3_conv1/Conv2DConv2D5functional_1/stack0_enc3_act0_relu/Relu:activations:0<functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2'
%functional_1/stack0_enc3_conv1/Conv2D?
5functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOpReadVariableOp>functional_1_stack0_enc3_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype027
5functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp?
&functional_1/stack0_enc3_conv1/BiasAddBiasAdd.functional_1/stack0_enc3_conv1/Conv2D:output:0=functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62(
&functional_1/stack0_enc3_conv1/BiasAdd?
'functional_1/stack0_enc3_act1_relu/ReluRelu/functional_1/stack0_enc3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62)
'functional_1/stack0_enc3_act1_relu/Relu?
*functional_1/stack0_enc4_last_pool/MaxPoolMaxPool5functional_1/stack0_enc3_act1_relu/Relu:activations:0*/
_output_shapes
:?????????  6*
ksize
*
paddingSAME*
strides
2,
*functional_1/stack0_enc4_last_pool/MaxPool?
Bfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOpReadVariableOpKfunctional_1_stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource*&
_output_shapes
:6Q*
dtype02D
Bfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?
3functional_1/stack0_enc5_middle_expand_conv0/Conv2DConv2D3functional_1/stack0_enc4_last_pool/MaxPool:output:0Jfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
25
3functional_1/stack0_enc5_middle_expand_conv0/Conv2D?
Cfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOpReadVariableOpLfunctional_1_stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02E
Cfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?
4functional_1/stack0_enc5_middle_expand_conv0/BiasAddBiasAdd<functional_1/stack0_enc5_middle_expand_conv0/Conv2D:output:0Kfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q26
4functional_1/stack0_enc5_middle_expand_conv0/BiasAdd?
5functional_1/stack0_enc5_middle_expand_act0_relu/ReluRelu=functional_1/stack0_enc5_middle_expand_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q27
5functional_1/stack0_enc5_middle_expand_act0_relu/Relu?
Dfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource*&
_output_shapes
:QQ*
dtype02F
Dfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
5functional_1/stack0_enc6_middle_contract_conv0/Conv2DConv2DCfunctional_1/stack0_enc5_middle_expand_act0_relu/Relu:activations:0Lfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
27
5functional_1/stack0_enc6_middle_contract_conv0/Conv2D?
Efunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOpReadVariableOpNfunctional_1_stack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02G
Efunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?
6functional_1/stack0_enc6_middle_contract_conv0/BiasAddBiasAdd>functional_1/stack0_enc6_middle_contract_conv0/Conv2D:output:0Mfunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q28
6functional_1/stack0_enc6_middle_contract_conv0/BiasAdd?
7functional_1/stack0_enc6_middle_contract_act0_relu/ReluRelu?functional_1/stack0_enc6_middle_contract_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q29
7functional_1/stack0_enc6_middle_contract_act0_relu/Relu?
8functional_1/stack0_dec0_s16_to_s8_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2:
8functional_1/stack0_dec0_s16_to_s8_interp_bilinear/Const?
:functional_1/stack0_dec0_s16_to_s8_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2<
:functional_1/stack0_dec0_s16_to_s8_interp_bilinear/Const_1?
6functional_1/stack0_dec0_s16_to_s8_interp_bilinear/mulMulAfunctional_1/stack0_dec0_s16_to_s8_interp_bilinear/Const:output:0Cfunctional_1/stack0_dec0_s16_to_s8_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:28
6functional_1/stack0_dec0_s16_to_s8_interp_bilinear/mul?
Hfunctional_1/stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinearResizeBilinearEfunctional_1/stack0_enc6_middle_contract_act0_relu/Relu:activations:0:functional_1/stack0_dec0_s16_to_s8_interp_bilinear/mul:z:0*
T0*/
_output_shapes
:?????????@@Q*
half_pixel_centers(2J
Hfunctional_1/stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear?
:functional_1/stack0_dec0_s16_to_s8_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2<
:functional_1/stack0_dec0_s16_to_s8_skip_concat/concat/axis?
5functional_1/stack0_dec0_s16_to_s8_skip_concat/concatConcatV25functional_1/stack0_enc3_act1_relu/Relu:activations:0Yfunctional_1/stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear:resized_images:0Cfunctional_1/stack0_dec0_s16_to_s8_skip_concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?27
5functional_1/stack0_dec0_s16_to_s8_skip_concat/concat?
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource*'
_output_shapes
:?6*
dtype02G
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?
6functional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2DConv2D>functional_1/stack0_dec0_s16_to_s8_skip_concat/concat:output:0Mfunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
28
6functional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D?
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpOfunctional_1_stack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02H
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?
7functional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAddBiasAdd?functional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D:output:0Nfunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@629
7functional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd?
=functional_1/stack0_dec0_s16_to_s8_refine_conv0_act_relu/ReluRelu@functional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62?
=functional_1/stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu?
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02G
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?
6functional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2DConv2DKfunctional_1/stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu:activations:0Mfunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
28
6functional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D?
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpOfunctional_1_stack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02H
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?
7functional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAddBiasAdd?functional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D:output:0Nfunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@629
7functional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd?
=functional_1/stack0_dec0_s16_to_s8_refine_conv1_act_relu/ReluRelu@functional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62?
=functional_1/stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu?
7functional_1/stack0_dec1_s8_to_s4_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   29
7functional_1/stack0_dec1_s8_to_s4_interp_bilinear/Const?
9functional_1/stack0_dec1_s8_to_s4_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2;
9functional_1/stack0_dec1_s8_to_s4_interp_bilinear/Const_1?
5functional_1/stack0_dec1_s8_to_s4_interp_bilinear/mulMul@functional_1/stack0_dec1_s8_to_s4_interp_bilinear/Const:output:0Bfunctional_1/stack0_dec1_s8_to_s4_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:27
5functional_1/stack0_dec1_s8_to_s4_interp_bilinear/mul?
Gfunctional_1/stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinearResizeBilinearKfunctional_1/stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu:activations:09functional_1/stack0_dec1_s8_to_s4_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????6*
half_pixel_centers(2I
Gfunctional_1/stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear?
9functional_1/stack0_dec1_s8_to_s4_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2;
9functional_1/stack0_dec1_s8_to_s4_skip_concat/concat/axis?
4functional_1/stack0_dec1_s8_to_s4_skip_concat/concatConcatV25functional_1/stack0_enc2_act1_relu/Relu:activations:0Xfunctional_1/stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear:resized_images:0Bfunctional_1/stack0_dec1_s8_to_s4_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????Z26
4functional_1/stack0_dec1_s8_to_s4_skip_concat/concat?
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:Z$*
dtype02F
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?
5functional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2DConv2D=functional_1/stack0_dec1_s8_to_s4_skip_concat/concat:output:0Lfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
27
5functional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D?
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02G
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?
6functional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAddBiasAdd>functional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D:output:0Mfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$28
6functional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd?
<functional_1/stack0_dec1_s8_to_s4_refine_conv0_act_relu/ReluRelu?functional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2>
<functional_1/stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu?
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02F
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?
5functional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2DConv2DJfunctional_1/stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu:activations:0Lfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
27
5functional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D?
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02G
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?
6functional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAddBiasAdd>functional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D:output:0Mfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$28
6functional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd?
<functional_1/stack0_dec1_s8_to_s4_refine_conv1_act_relu/ReluRelu?functional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2>
<functional_1/stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu?
7functional_1/stack0_dec2_s4_to_s2_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   29
7functional_1/stack0_dec2_s4_to_s2_interp_bilinear/Const?
9functional_1/stack0_dec2_s4_to_s2_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2;
9functional_1/stack0_dec2_s4_to_s2_interp_bilinear/Const_1?
5functional_1/stack0_dec2_s4_to_s2_interp_bilinear/mulMul@functional_1/stack0_dec2_s4_to_s2_interp_bilinear/Const:output:0Bfunctional_1/stack0_dec2_s4_to_s2_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:27
5functional_1/stack0_dec2_s4_to_s2_interp_bilinear/mul?
Gfunctional_1/stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinearResizeBilinearJfunctional_1/stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu:activations:09functional_1/stack0_dec2_s4_to_s2_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????$*
half_pixel_centers(2I
Gfunctional_1/stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear?
9functional_1/stack0_dec2_s4_to_s2_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2;
9functional_1/stack0_dec2_s4_to_s2_skip_concat/concat/axis?
4functional_1/stack0_dec2_s4_to_s2_skip_concat/concatConcatV25functional_1/stack0_enc1_act1_relu/Relu:activations:0Xfunctional_1/stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear:resized_images:0Bfunctional_1/stack0_dec2_s4_to_s2_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????<26
4functional_1/stack0_dec2_s4_to_s2_skip_concat/concat?
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02F
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?
5functional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2DConv2D=functional_1/stack0_dec2_s4_to_s2_skip_concat/concat:output:0Lfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
27
5functional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D?
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?
6functional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAddBiasAdd>functional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D:output:0Mfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????28
6functional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd?
<functional_1/stack0_dec2_s4_to_s2_refine_conv0_act_relu/ReluRelu?functional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2>
<functional_1/stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu?
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOpReadVariableOpMfunctional_1_stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02F
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?
5functional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2DConv2DJfunctional_1/stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu:activations:0Lfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
27
5functional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D?
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpNfunctional_1_stack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02G
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?
6functional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAddBiasAdd>functional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D:output:0Mfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????28
6functional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd?
<functional_1/stack0_dec2_s4_to_s2_refine_conv1_act_relu/ReluRelu?functional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2>
<functional_1/stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu?
7functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOpReadVariableOp@functional_1_centroidconfmapshead_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOp?
(functional_1/CentroidConfmapsHead/Conv2DConv2DJfunctional_1/stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu:activations:0?functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(functional_1/CentroidConfmapsHead/Conv2D?
8functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOpReadVariableOpAfunctional_1_centroidconfmapshead_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp?
)functional_1/CentroidConfmapsHead/BiasAddBiasAdd1functional_1/CentroidConfmapsHead/Conv2D:output:0@functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)functional_1/CentroidConfmapsHead/BiasAdd?
IdentityIdentity2functional_1/CentroidConfmapsHead/BiasAdd:output:09^functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp8^functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOpG^functional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOpF^functional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOpG^functional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOpF^functional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOpF^functional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOpE^functional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOpF^functional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOpE^functional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOpF^functional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOpE^functional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOpF^functional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOpE^functional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp6^functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp5^functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp6^functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp5^functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp6^functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp5^functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp6^functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp5^functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp6^functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp5^functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp6^functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp5^functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp6^functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp5^functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp6^functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp5^functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOpD^functional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOpC^functional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOpF^functional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOpE^functional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2t
8functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp8functional_1/CentroidConfmapsHead/BiasAdd/ReadVariableOp2r
7functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOp7functional_1/CentroidConfmapsHead/Conv2D/ReadVariableOp2?
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOpFfunctional_1/stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp2?
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOpEfunctional_1/stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp2?
Ffunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOpFfunctional_1/stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp2?
Efunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOpEfunctional_1/stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp2?
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOpEfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp2?
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOpDfunctional_1/stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp2?
Efunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOpEfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp2?
Dfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOpDfunctional_1/stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp2?
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOpEfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp2?
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOpDfunctional_1/stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp2?
Efunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOpEfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp2?
Dfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOpDfunctional_1/stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp5functional_1/stack0_enc0_conv0/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp4functional_1/stack0_enc0_conv0/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp5functional_1/stack0_enc0_conv1/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp4functional_1/stack0_enc0_conv1/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp5functional_1/stack0_enc1_conv0/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp4functional_1/stack0_enc1_conv0/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp5functional_1/stack0_enc1_conv1/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp4functional_1/stack0_enc1_conv1/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp5functional_1/stack0_enc2_conv0/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp4functional_1/stack0_enc2_conv0/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp5functional_1/stack0_enc2_conv1/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp4functional_1/stack0_enc2_conv1/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp5functional_1/stack0_enc3_conv0/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp4functional_1/stack0_enc3_conv0/Conv2D/ReadVariableOp2n
5functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp5functional_1/stack0_enc3_conv1/BiasAdd/ReadVariableOp2l
4functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOp4functional_1/stack0_enc3_conv1/Conv2D/ReadVariableOp2?
Cfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOpCfunctional_1/stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp2?
Bfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOpBfunctional_1/stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp2?
Efunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOpEfunctional_1/stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp2?
Dfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOpDfunctional_1/stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
l
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_18847

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_18870

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc3_act0_relu_layer_call_fn_20733

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_188472
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc2_act1_relu_layer_call_fn_20704

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_188232
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
?
B__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_fn_20847

inputs"
unknown:?6
	unknown_0:6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_189392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_18823

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_20928

inputs8
conv2d_readvariableop_resource:Z$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:Z$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????Z: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????Z
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_21009

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_18789

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
_
C__inference_stack0_enc5_middle_expand_act0_relu_layer_call_fn_20791

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *g
fbR`
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_188942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
?	
#__inference_signature_wrapper_20101	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:$
	unknown_8:$#
	unknown_9:$$

unknown_10:$$

unknown_11:$6

unknown_12:6$

unknown_13:66

unknown_14:6$

unknown_15:6Q

unknown_16:Q$

unknown_17:QQ

unknown_18:Q%

unknown_19:?6

unknown_20:6$

unknown_21:66

unknown_22:6$

unknown_23:Z$

unknown_24:$$

unknown_25:$$

unknown_26:$$

unknown_27:<

unknown_28:$

unknown_29:

unknown_30:$

unknown_31:

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__wrapped_model_185732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_19029

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_18695

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_20699

inputs8
conv2d_readvariableop_resource:$$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
z
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_20796

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  Q2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc1_conv1_layer_call_fn_20631

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_187652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
f
J__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_fn_21004

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190622
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
L
0__inference_stack0_enc3_pool_layer_call_fn_18609

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_186032
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_20670

inputs8
conv2d_readvariableop_resource:$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_20680

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_19652

inputs1
stack0_enc0_conv0_19540:%
stack0_enc0_conv0_19542:1
stack0_enc0_conv1_19546:%
stack0_enc0_conv1_19548:1
stack0_enc1_conv0_19553:%
stack0_enc1_conv0_19555:1
stack0_enc1_conv1_19559:%
stack0_enc1_conv1_19561:1
stack0_enc2_conv0_19566:$%
stack0_enc2_conv0_19568:$1
stack0_enc2_conv1_19572:$$%
stack0_enc2_conv1_19574:$1
stack0_enc3_conv0_19579:$6%
stack0_enc3_conv0_19581:61
stack0_enc3_conv1_19585:66%
stack0_enc3_conv1_19587:6?
%stack0_enc5_middle_expand_conv0_19592:6Q3
%stack0_enc5_middle_expand_conv0_19594:QA
'stack0_enc6_middle_contract_conv0_19598:QQ5
'stack0_enc6_middle_contract_conv0_19600:QC
(stack0_dec0_s16_to_s8_refine_conv0_19606:?66
(stack0_dec0_s16_to_s8_refine_conv0_19608:6B
(stack0_dec0_s16_to_s8_refine_conv1_19612:666
(stack0_dec0_s16_to_s8_refine_conv1_19614:6A
'stack0_dec1_s8_to_s4_refine_conv0_19620:Z$5
'stack0_dec1_s8_to_s4_refine_conv0_19622:$A
'stack0_dec1_s8_to_s4_refine_conv1_19626:$$5
'stack0_dec1_s8_to_s4_refine_conv1_19628:$A
'stack0_dec2_s4_to_s2_refine_conv0_19634:<5
'stack0_dec2_s4_to_s2_refine_conv0_19636:A
'stack0_dec2_s4_to_s2_refine_conv1_19640:5
'stack0_dec2_s4_to_s2_refine_conv1_19642:4
centroidconfmapshead_19646:(
centroidconfmapshead_19648:
identity??,CentroidConfmapsHead/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?)stack0_enc0_conv0/StatefulPartitionedCall?)stack0_enc0_conv1/StatefulPartitionedCall?)stack0_enc1_conv0/StatefulPartitionedCall?)stack0_enc1_conv1/StatefulPartitionedCall?)stack0_enc2_conv0/StatefulPartitionedCall?)stack0_enc2_conv1/StatefulPartitionedCall?)stack0_enc3_conv0/StatefulPartitionedCall?)stack0_enc3_conv1/StatefulPartitionedCall?7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
)stack0_enc0_conv0/StatefulPartitionedCallStatefulPartitionedCallinputsstack0_enc0_conv0_19540stack0_enc0_conv0_19542*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_186952+
)stack0_enc0_conv0/StatefulPartitionedCall?
%stack0_enc0_act0_relu/PartitionedCallPartitionedCall2stack0_enc0_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_187062'
%stack0_enc0_act0_relu/PartitionedCall?
)stack0_enc0_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc0_act0_relu/PartitionedCall:output:0stack0_enc0_conv1_19546stack0_enc0_conv1_19548*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_187182+
)stack0_enc0_conv1/StatefulPartitionedCall?
%stack0_enc0_act1_relu/PartitionedCallPartitionedCall2stack0_enc0_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_187292'
%stack0_enc0_act1_relu/PartitionedCall?
 stack0_enc1_pool/PartitionedCallPartitionedCall.stack0_enc0_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_185792"
 stack0_enc1_pool/PartitionedCall?
)stack0_enc1_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc1_pool/PartitionedCall:output:0stack0_enc1_conv0_19553stack0_enc1_conv0_19555*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_187422+
)stack0_enc1_conv0/StatefulPartitionedCall?
%stack0_enc1_act0_relu/PartitionedCallPartitionedCall2stack0_enc1_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_187532'
%stack0_enc1_act0_relu/PartitionedCall?
)stack0_enc1_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc1_act0_relu/PartitionedCall:output:0stack0_enc1_conv1_19559stack0_enc1_conv1_19561*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_187652+
)stack0_enc1_conv1/StatefulPartitionedCall?
%stack0_enc1_act1_relu/PartitionedCallPartitionedCall2stack0_enc1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_187762'
%stack0_enc1_act1_relu/PartitionedCall?
 stack0_enc2_pool/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_185912"
 stack0_enc2_pool/PartitionedCall?
)stack0_enc2_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc2_pool/PartitionedCall:output:0stack0_enc2_conv0_19566stack0_enc2_conv0_19568*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_187892+
)stack0_enc2_conv0/StatefulPartitionedCall?
%stack0_enc2_act0_relu/PartitionedCallPartitionedCall2stack0_enc2_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_188002'
%stack0_enc2_act0_relu/PartitionedCall?
)stack0_enc2_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc2_act0_relu/PartitionedCall:output:0stack0_enc2_conv1_19572stack0_enc2_conv1_19574*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_188122+
)stack0_enc2_conv1/StatefulPartitionedCall?
%stack0_enc2_act1_relu/PartitionedCallPartitionedCall2stack0_enc2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_188232'
%stack0_enc2_act1_relu/PartitionedCall?
 stack0_enc3_pool/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_186032"
 stack0_enc3_pool/PartitionedCall?
)stack0_enc3_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc3_pool/PartitionedCall:output:0stack0_enc3_conv0_19579stack0_enc3_conv0_19581*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_188362+
)stack0_enc3_conv0/StatefulPartitionedCall?
%stack0_enc3_act0_relu/PartitionedCallPartitionedCall2stack0_enc3_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_188472'
%stack0_enc3_act0_relu/PartitionedCall?
)stack0_enc3_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc3_act0_relu/PartitionedCall:output:0stack0_enc3_conv1_19585stack0_enc3_conv1_19587*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_188592+
)stack0_enc3_conv1/StatefulPartitionedCall?
%stack0_enc3_act1_relu/PartitionedCallPartitionedCall2stack0_enc3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_188702'
%stack0_enc3_act1_relu/PartitionedCall?
%stack0_enc4_last_pool/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_186152'
%stack0_enc4_last_pool/PartitionedCall?
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc4_last_pool/PartitionedCall:output:0%stack0_enc5_middle_expand_conv0_19592%stack0_enc5_middle_expand_conv0_19594*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_1888329
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?
3stack0_enc5_middle_expand_act0_relu/PartitionedCallPartitionedCall@stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *g
fbR`
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_1889425
3stack0_enc5_middle_expand_act0_relu/PartitionedCall?
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCallStatefulPartitionedCall<stack0_enc5_middle_expand_act0_relu/PartitionedCall:output:0'stack0_enc6_middle_contract_conv0_19598'stack0_enc6_middle_contract_conv0_19600*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_189062;
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
5stack0_enc6_middle_contract_act0_relu/PartitionedCallPartitionedCallBstack0_enc6_middle_contract_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_1891727
5stack0_enc6_middle_contract_act0_relu/PartitionedCall?
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCallPartitionedCall>stack0_enc6_middle_contract_act0_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_1863427
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall?
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0>stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_1892723
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall:stack0_dec0_s16_to_s8_skip_concat/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv0_19606(stack0_dec0_s16_to_s8_refine_conv0_19608*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_189392<
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_189502=
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallDstack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv1_19612(stack0_dec0_s16_to_s8_refine_conv1_19614*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_189622<
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_189732=
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall?
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCallPartitionedCallDstack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_1865326
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall?
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0=stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_1898322
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec1_s8_to_s4_skip_concat/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv0_19620'stack0_dec1_s8_to_s4_refine_conv0_19622*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_189952;
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190062<
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv1_19626'stack0_dec1_s8_to_s4_refine_conv1_19628*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_190182;
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190292<
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall?
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCallPartitionedCallCstack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_1867226
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall?
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0=stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_1903922
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec2_s4_to_s2_skip_concat/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv0_19634'stack0_dec2_s4_to_s2_refine_conv0_19636*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_190512;
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190622<
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv1_19640'stack0_dec2_s4_to_s2_refine_conv1_19642*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_190742;
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190852<
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall?
,CentroidConfmapsHead/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall:output:0centroidconfmapshead_19646centroidconfmapshead_19648*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_190972.
,CentroidConfmapsHead/StatefulPartitionedCall?
IdentityIdentity5CentroidConfmapsHead/StatefulPartitionedCall:output:0-^CentroidConfmapsHead/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall*^stack0_enc0_conv0/StatefulPartitionedCall*^stack0_enc0_conv1/StatefulPartitionedCall*^stack0_enc1_conv0/StatefulPartitionedCall*^stack0_enc1_conv1/StatefulPartitionedCall*^stack0_enc2_conv0/StatefulPartitionedCall*^stack0_enc2_conv1/StatefulPartitionedCall*^stack0_enc3_conv0/StatefulPartitionedCall*^stack0_enc3_conv1/StatefulPartitionedCall8^stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:^stack0_enc6_middle_contract_conv0/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,CentroidConfmapsHead/StatefulPartitionedCall,CentroidConfmapsHead/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall2V
)stack0_enc0_conv0/StatefulPartitionedCall)stack0_enc0_conv0/StatefulPartitionedCall2V
)stack0_enc0_conv1/StatefulPartitionedCall)stack0_enc0_conv1/StatefulPartitionedCall2V
)stack0_enc1_conv0/StatefulPartitionedCall)stack0_enc1_conv0/StatefulPartitionedCall2V
)stack0_enc1_conv1/StatefulPartitionedCall)stack0_enc1_conv1/StatefulPartitionedCall2V
)stack0_enc2_conv0/StatefulPartitionedCall)stack0_enc2_conv0/StatefulPartitionedCall2V
)stack0_enc2_conv1/StatefulPartitionedCall)stack0_enc2_conv1/StatefulPartitionedCall2V
)stack0_enc3_conv0/StatefulPartitionedCall)stack0_enc3_conv0/StatefulPartitionedCall2V
)stack0_enc3_conv1/StatefulPartitionedCall)stack0_enc3_conv1/StatefulPartitionedCall2r
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall2v
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_19911	
input1
stack0_enc0_conv0_19799:%
stack0_enc0_conv0_19801:1
stack0_enc0_conv1_19805:%
stack0_enc0_conv1_19807:1
stack0_enc1_conv0_19812:%
stack0_enc1_conv0_19814:1
stack0_enc1_conv1_19818:%
stack0_enc1_conv1_19820:1
stack0_enc2_conv0_19825:$%
stack0_enc2_conv0_19827:$1
stack0_enc2_conv1_19831:$$%
stack0_enc2_conv1_19833:$1
stack0_enc3_conv0_19838:$6%
stack0_enc3_conv0_19840:61
stack0_enc3_conv1_19844:66%
stack0_enc3_conv1_19846:6?
%stack0_enc5_middle_expand_conv0_19851:6Q3
%stack0_enc5_middle_expand_conv0_19853:QA
'stack0_enc6_middle_contract_conv0_19857:QQ5
'stack0_enc6_middle_contract_conv0_19859:QC
(stack0_dec0_s16_to_s8_refine_conv0_19865:?66
(stack0_dec0_s16_to_s8_refine_conv0_19867:6B
(stack0_dec0_s16_to_s8_refine_conv1_19871:666
(stack0_dec0_s16_to_s8_refine_conv1_19873:6A
'stack0_dec1_s8_to_s4_refine_conv0_19879:Z$5
'stack0_dec1_s8_to_s4_refine_conv0_19881:$A
'stack0_dec1_s8_to_s4_refine_conv1_19885:$$5
'stack0_dec1_s8_to_s4_refine_conv1_19887:$A
'stack0_dec2_s4_to_s2_refine_conv0_19893:<5
'stack0_dec2_s4_to_s2_refine_conv0_19895:A
'stack0_dec2_s4_to_s2_refine_conv1_19899:5
'stack0_dec2_s4_to_s2_refine_conv1_19901:4
centroidconfmapshead_19905:(
centroidconfmapshead_19907:
identity??,CentroidConfmapsHead/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?)stack0_enc0_conv0/StatefulPartitionedCall?)stack0_enc0_conv1/StatefulPartitionedCall?)stack0_enc1_conv0/StatefulPartitionedCall?)stack0_enc1_conv1/StatefulPartitionedCall?)stack0_enc2_conv0/StatefulPartitionedCall?)stack0_enc2_conv1/StatefulPartitionedCall?)stack0_enc3_conv0/StatefulPartitionedCall?)stack0_enc3_conv1/StatefulPartitionedCall?7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
)stack0_enc0_conv0/StatefulPartitionedCallStatefulPartitionedCallinputstack0_enc0_conv0_19799stack0_enc0_conv0_19801*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_186952+
)stack0_enc0_conv0/StatefulPartitionedCall?
%stack0_enc0_act0_relu/PartitionedCallPartitionedCall2stack0_enc0_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_187062'
%stack0_enc0_act0_relu/PartitionedCall?
)stack0_enc0_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc0_act0_relu/PartitionedCall:output:0stack0_enc0_conv1_19805stack0_enc0_conv1_19807*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_187182+
)stack0_enc0_conv1/StatefulPartitionedCall?
%stack0_enc0_act1_relu/PartitionedCallPartitionedCall2stack0_enc0_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_187292'
%stack0_enc0_act1_relu/PartitionedCall?
 stack0_enc1_pool/PartitionedCallPartitionedCall.stack0_enc0_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_185792"
 stack0_enc1_pool/PartitionedCall?
)stack0_enc1_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc1_pool/PartitionedCall:output:0stack0_enc1_conv0_19812stack0_enc1_conv0_19814*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_187422+
)stack0_enc1_conv0/StatefulPartitionedCall?
%stack0_enc1_act0_relu/PartitionedCallPartitionedCall2stack0_enc1_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_187532'
%stack0_enc1_act0_relu/PartitionedCall?
)stack0_enc1_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc1_act0_relu/PartitionedCall:output:0stack0_enc1_conv1_19818stack0_enc1_conv1_19820*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_187652+
)stack0_enc1_conv1/StatefulPartitionedCall?
%stack0_enc1_act1_relu/PartitionedCallPartitionedCall2stack0_enc1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_187762'
%stack0_enc1_act1_relu/PartitionedCall?
 stack0_enc2_pool/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_185912"
 stack0_enc2_pool/PartitionedCall?
)stack0_enc2_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc2_pool/PartitionedCall:output:0stack0_enc2_conv0_19825stack0_enc2_conv0_19827*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_187892+
)stack0_enc2_conv0/StatefulPartitionedCall?
%stack0_enc2_act0_relu/PartitionedCallPartitionedCall2stack0_enc2_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_188002'
%stack0_enc2_act0_relu/PartitionedCall?
)stack0_enc2_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc2_act0_relu/PartitionedCall:output:0stack0_enc2_conv1_19831stack0_enc2_conv1_19833*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_188122+
)stack0_enc2_conv1/StatefulPartitionedCall?
%stack0_enc2_act1_relu/PartitionedCallPartitionedCall2stack0_enc2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_188232'
%stack0_enc2_act1_relu/PartitionedCall?
 stack0_enc3_pool/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_186032"
 stack0_enc3_pool/PartitionedCall?
)stack0_enc3_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc3_pool/PartitionedCall:output:0stack0_enc3_conv0_19838stack0_enc3_conv0_19840*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_188362+
)stack0_enc3_conv0/StatefulPartitionedCall?
%stack0_enc3_act0_relu/PartitionedCallPartitionedCall2stack0_enc3_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_188472'
%stack0_enc3_act0_relu/PartitionedCall?
)stack0_enc3_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc3_act0_relu/PartitionedCall:output:0stack0_enc3_conv1_19844stack0_enc3_conv1_19846*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_188592+
)stack0_enc3_conv1/StatefulPartitionedCall?
%stack0_enc3_act1_relu/PartitionedCallPartitionedCall2stack0_enc3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_188702'
%stack0_enc3_act1_relu/PartitionedCall?
%stack0_enc4_last_pool/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_186152'
%stack0_enc4_last_pool/PartitionedCall?
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc4_last_pool/PartitionedCall:output:0%stack0_enc5_middle_expand_conv0_19851%stack0_enc5_middle_expand_conv0_19853*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_1888329
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?
3stack0_enc5_middle_expand_act0_relu/PartitionedCallPartitionedCall@stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *g
fbR`
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_1889425
3stack0_enc5_middle_expand_act0_relu/PartitionedCall?
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCallStatefulPartitionedCall<stack0_enc5_middle_expand_act0_relu/PartitionedCall:output:0'stack0_enc6_middle_contract_conv0_19857'stack0_enc6_middle_contract_conv0_19859*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_189062;
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
5stack0_enc6_middle_contract_act0_relu/PartitionedCallPartitionedCallBstack0_enc6_middle_contract_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_1891727
5stack0_enc6_middle_contract_act0_relu/PartitionedCall?
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCallPartitionedCall>stack0_enc6_middle_contract_act0_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_1863427
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall?
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0>stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_1892723
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall:stack0_dec0_s16_to_s8_skip_concat/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv0_19865(stack0_dec0_s16_to_s8_refine_conv0_19867*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_189392<
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_189502=
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallDstack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv1_19871(stack0_dec0_s16_to_s8_refine_conv1_19873*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_189622<
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_189732=
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall?
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCallPartitionedCallDstack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_1865326
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall?
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0=stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_1898322
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec1_s8_to_s4_skip_concat/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv0_19879'stack0_dec1_s8_to_s4_refine_conv0_19881*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_189952;
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190062<
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv1_19885'stack0_dec1_s8_to_s4_refine_conv1_19887*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_190182;
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190292<
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall?
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCallPartitionedCallCstack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_1867226
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall?
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0=stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_1903922
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec2_s4_to_s2_skip_concat/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv0_19893'stack0_dec2_s4_to_s2_refine_conv0_19895*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_190512;
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190622<
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv1_19899'stack0_dec2_s4_to_s2_refine_conv1_19901*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_190742;
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190852<
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall?
,CentroidConfmapsHead/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall:output:0centroidconfmapshead_19905centroidconfmapshead_19907*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_190972.
,CentroidConfmapsHead/StatefulPartitionedCall?
IdentityIdentity5CentroidConfmapsHead/StatefulPartitionedCall:output:0-^CentroidConfmapsHead/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall*^stack0_enc0_conv0/StatefulPartitionedCall*^stack0_enc0_conv1/StatefulPartitionedCall*^stack0_enc1_conv0/StatefulPartitionedCall*^stack0_enc1_conv1/StatefulPartitionedCall*^stack0_enc2_conv0/StatefulPartitionedCall*^stack0_enc2_conv1/StatefulPartitionedCall*^stack0_enc3_conv0/StatefulPartitionedCall*^stack0_enc3_conv1/StatefulPartitionedCall8^stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:^stack0_enc6_middle_contract_conv0/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,CentroidConfmapsHead/StatefulPartitionedCall,CentroidConfmapsHead/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall2V
)stack0_enc0_conv0/StatefulPartitionedCall)stack0_enc0_conv0/StatefulPartitionedCall2V
)stack0_enc0_conv1/StatefulPartitionedCall)stack0_enc0_conv1/StatefulPartitionedCall2V
)stack0_enc1_conv0/StatefulPartitionedCall)stack0_enc1_conv0/StatefulPartitionedCall2V
)stack0_enc1_conv1/StatefulPartitionedCall)stack0_enc1_conv1/StatefulPartitionedCall2V
)stack0_enc2_conv0/StatefulPartitionedCall)stack0_enc2_conv0/StatefulPartitionedCall2V
)stack0_enc2_conv1/StatefulPartitionedCall)stack0_enc2_conv1/StatefulPartitionedCall2V
)stack0_enc3_conv0/StatefulPartitionedCall)stack0_enc3_conv0/StatefulPartitionedCall2V
)stack0_enc3_conv1/StatefulPartitionedCall)stack0_enc3_conv1/StatefulPartitionedCall2r
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall2v
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?

?
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_18962

inputs8
conv2d_readvariableop_resource:66-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
?
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20867

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
a
E__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_fn_18640

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_186342
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc0_conv0_layer_call_fn_20544

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_186952
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_18859

inputs8
conv2d_readvariableop_resource:66-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_20999

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????<
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc3_conv1_layer_call_fn_20747

inputs!
unknown:66
	unknown_0:6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_188592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_19062

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
A__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_fn_20989

inputs!
unknown:<
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_190512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????<
 
_user_specified_nameinputs
?
?
A__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_fn_20947

inputs!
unknown:$$
	unknown_0:$
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_190182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
|
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_18917

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  Q2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
{
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_18672

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeBilinearResizeBilinearinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeBilinear?
IdentityIdentity&resize/ResizeBilinear:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_18927

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?2
concatl
IdentityIdentityconcat:output:0*
T0*0
_output_shapes
:?????????@@?2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*[
_input_shapesJ
H:?????????@@6:+???????????????????????????Q:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????Q
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_20583

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_20909
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????Z2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????$:+???????????????????????????6:[ W
1
_output_shapes
:???????????$
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????6
"
_user_specified_name
inputs/1
?

?
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_18939

inputs9
conv2d_readvariableop_resource:?6-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?6*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????@@?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????@@?
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_18753

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_20622

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_20651

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
`
D__inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_fn_18659

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_186532
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20967

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_19006

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
l
@__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_fn_20973
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_190392
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????<2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????:+???????????????????????????$:[ W
1
_output_shapes
:???????????
"
_user_specified_name
inputs/0:kg
A
_output_shapes/
-:+???????????????????????????$
"
_user_specified_name
inputs/1
?
?
?__inference_stack0_enc5_middle_expand_conv0_layer_call_fn_20776

inputs!
unknown:6Q
	unknown_0:Q
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_188832
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  6: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  6
 
_user_specified_nameinputs
?
a
E__inference_stack0_enc6_middle_contract_act0_relu_layer_call_fn_20820

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_189172
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_20728

inputs8
conv2d_readvariableop_resource:$6-
biasadd_readvariableop_resource:6
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$6*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:6*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@@$
 
_user_specified_nameinputs
?
?
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_18950

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????@@62
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc1_act1_relu_layer_call_fn_20646

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_187762
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
B__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_fn_20876

inputs!
unknown:66
	unknown_0:6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_189622
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@6: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
f
J__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_fn_20933

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190062
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?

?
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_20641

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_21028

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_fn_20891

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_189732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@@6:W S
/
_output_shapes
:?????????@@6
 
_user_specified_nameinputs
?
?
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_18983

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis?
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????Z2
concatm
IdentityIdentityconcat:output:0*
T0*1
_output_shapes
:???????????Z2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*]
_input_shapesL
J:???????????$:+???????????????????????????6:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs:ie
A
_output_shapes/
-:+???????????????????????????6
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_19051

inputs8
conv2d_readvariableop_resource:<-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:<*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????<
 
_user_specified_nameinputs
?
z
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_18894

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:?????????  Q2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????  Q:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc1_conv0_layer_call_fn_20602

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_187422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
܏
?"
G__inference_functional_1_layer_call_and_return_conditional_losses_20391

inputsJ
0stack0_enc0_conv0_conv2d_readvariableop_resource:?
1stack0_enc0_conv0_biasadd_readvariableop_resource:J
0stack0_enc0_conv1_conv2d_readvariableop_resource:?
1stack0_enc0_conv1_biasadd_readvariableop_resource:J
0stack0_enc1_conv0_conv2d_readvariableop_resource:?
1stack0_enc1_conv0_biasadd_readvariableop_resource:J
0stack0_enc1_conv1_conv2d_readvariableop_resource:?
1stack0_enc1_conv1_biasadd_readvariableop_resource:J
0stack0_enc2_conv0_conv2d_readvariableop_resource:$?
1stack0_enc2_conv0_biasadd_readvariableop_resource:$J
0stack0_enc2_conv1_conv2d_readvariableop_resource:$$?
1stack0_enc2_conv1_biasadd_readvariableop_resource:$J
0stack0_enc3_conv0_conv2d_readvariableop_resource:$6?
1stack0_enc3_conv0_biasadd_readvariableop_resource:6J
0stack0_enc3_conv1_conv2d_readvariableop_resource:66?
1stack0_enc3_conv1_biasadd_readvariableop_resource:6X
>stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource:6QM
?stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource:QZ
@stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource:QQO
Astack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource:Q\
Astack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource:?6P
Bstack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource:6[
Astack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource:66P
Bstack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource:6Z
@stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource:Z$O
Astack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource:$Z
@stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource:$$O
Astack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource:$Z
@stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource:<O
Astack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource:Z
@stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource:O
Astack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource:M
3centroidconfmapshead_conv2d_readvariableop_resource:B
4centroidconfmapshead_biasadd_readvariableop_resource:
identity??+CentroidConfmapsHead/BiasAdd/ReadVariableOp?*CentroidConfmapsHead/Conv2D/ReadVariableOp?9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?(stack0_enc0_conv0/BiasAdd/ReadVariableOp?'stack0_enc0_conv0/Conv2D/ReadVariableOp?(stack0_enc0_conv1/BiasAdd/ReadVariableOp?'stack0_enc0_conv1/Conv2D/ReadVariableOp?(stack0_enc1_conv0/BiasAdd/ReadVariableOp?'stack0_enc1_conv0/Conv2D/ReadVariableOp?(stack0_enc1_conv1/BiasAdd/ReadVariableOp?'stack0_enc1_conv1/Conv2D/ReadVariableOp?(stack0_enc2_conv0/BiasAdd/ReadVariableOp?'stack0_enc2_conv0/Conv2D/ReadVariableOp?(stack0_enc2_conv1/BiasAdd/ReadVariableOp?'stack0_enc2_conv1/Conv2D/ReadVariableOp?(stack0_enc3_conv0/BiasAdd/ReadVariableOp?'stack0_enc3_conv0/Conv2D/ReadVariableOp?(stack0_enc3_conv1/BiasAdd/ReadVariableOp?'stack0_enc3_conv1/Conv2D/ReadVariableOp?6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
'stack0_enc0_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc0_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc0_conv0/Conv2D/ReadVariableOp?
stack0_enc0_conv0/Conv2DConv2Dinputs/stack0_enc0_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc0_conv0/Conv2D?
(stack0_enc0_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc0_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc0_conv0/BiasAdd/ReadVariableOp?
stack0_enc0_conv0/BiasAddBiasAdd!stack0_enc0_conv0/Conv2D:output:00stack0_enc0_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_conv0/BiasAdd?
stack0_enc0_act0_relu/ReluRelu"stack0_enc0_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_act0_relu/Relu?
'stack0_enc0_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc0_conv1/Conv2D/ReadVariableOp?
stack0_enc0_conv1/Conv2DConv2D(stack0_enc0_act0_relu/Relu:activations:0/stack0_enc0_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc0_conv1/Conv2D?
(stack0_enc0_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc0_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc0_conv1/BiasAdd/ReadVariableOp?
stack0_enc0_conv1/BiasAddBiasAdd!stack0_enc0_conv1/Conv2D:output:00stack0_enc0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_conv1/BiasAdd?
stack0_enc0_act1_relu/ReluRelu"stack0_enc0_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_act1_relu/Relu?
stack0_enc1_pool/MaxPoolMaxPool(stack0_enc0_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
stack0_enc1_pool/MaxPool?
'stack0_enc1_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc1_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc1_conv0/Conv2D/ReadVariableOp?
stack0_enc1_conv0/Conv2DConv2D!stack0_enc1_pool/MaxPool:output:0/stack0_enc1_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc1_conv0/Conv2D?
(stack0_enc1_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc1_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc1_conv0/BiasAdd/ReadVariableOp?
stack0_enc1_conv0/BiasAddBiasAdd!stack0_enc1_conv0/Conv2D:output:00stack0_enc1_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_conv0/BiasAdd?
stack0_enc1_act0_relu/ReluRelu"stack0_enc1_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_act0_relu/Relu?
'stack0_enc1_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc1_conv1/Conv2D/ReadVariableOp?
stack0_enc1_conv1/Conv2DConv2D(stack0_enc1_act0_relu/Relu:activations:0/stack0_enc1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc1_conv1/Conv2D?
(stack0_enc1_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc1_conv1/BiasAdd/ReadVariableOp?
stack0_enc1_conv1/BiasAddBiasAdd!stack0_enc1_conv1/Conv2D:output:00stack0_enc1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_conv1/BiasAdd?
stack0_enc1_act1_relu/ReluRelu"stack0_enc1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_act1_relu/Relu?
stack0_enc2_pool/MaxPoolMaxPool(stack0_enc1_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
stack0_enc2_pool/MaxPool?
'stack0_enc2_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc2_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02)
'stack0_enc2_conv0/Conv2D/ReadVariableOp?
stack0_enc2_conv0/Conv2DConv2D!stack0_enc2_pool/MaxPool:output:0/stack0_enc2_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
stack0_enc2_conv0/Conv2D?
(stack0_enc2_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc2_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02*
(stack0_enc2_conv0/BiasAdd/ReadVariableOp?
stack0_enc2_conv0/BiasAddBiasAdd!stack0_enc2_conv0/Conv2D:output:00stack0_enc2_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_conv0/BiasAdd?
stack0_enc2_act0_relu/ReluRelu"stack0_enc2_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_act0_relu/Relu?
'stack0_enc2_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02)
'stack0_enc2_conv1/Conv2D/ReadVariableOp?
stack0_enc2_conv1/Conv2DConv2D(stack0_enc2_act0_relu/Relu:activations:0/stack0_enc2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
stack0_enc2_conv1/Conv2D?
(stack0_enc2_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc2_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02*
(stack0_enc2_conv1/BiasAdd/ReadVariableOp?
stack0_enc2_conv1/BiasAddBiasAdd!stack0_enc2_conv1/Conv2D:output:00stack0_enc2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_conv1/BiasAdd?
stack0_enc2_act1_relu/ReluRelu"stack0_enc2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_act1_relu/Relu?
stack0_enc3_pool/MaxPoolMaxPool(stack0_enc2_act1_relu/Relu:activations:0*/
_output_shapes
:?????????@@$*
ksize
*
paddingSAME*
strides
2
stack0_enc3_pool/MaxPool?
'stack0_enc3_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc3_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$6*
dtype02)
'stack0_enc3_conv0/Conv2D/ReadVariableOp?
stack0_enc3_conv0/Conv2DConv2D!stack0_enc3_pool/MaxPool:output:0/stack0_enc3_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
stack0_enc3_conv0/Conv2D?
(stack0_enc3_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc3_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02*
(stack0_enc3_conv0/BiasAdd/ReadVariableOp?
stack0_enc3_conv0/BiasAddBiasAdd!stack0_enc3_conv0/Conv2D:output:00stack0_enc3_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_conv0/BiasAdd?
stack0_enc3_act0_relu/ReluRelu"stack0_enc3_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_act0_relu/Relu?
'stack0_enc3_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02)
'stack0_enc3_conv1/Conv2D/ReadVariableOp?
stack0_enc3_conv1/Conv2DConv2D(stack0_enc3_act0_relu/Relu:activations:0/stack0_enc3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
stack0_enc3_conv1/Conv2D?
(stack0_enc3_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc3_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02*
(stack0_enc3_conv1/BiasAdd/ReadVariableOp?
stack0_enc3_conv1/BiasAddBiasAdd!stack0_enc3_conv1/Conv2D:output:00stack0_enc3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_conv1/BiasAdd?
stack0_enc3_act1_relu/ReluRelu"stack0_enc3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_act1_relu/Relu?
stack0_enc4_last_pool/MaxPoolMaxPool(stack0_enc3_act1_relu/Relu:activations:0*/
_output_shapes
:?????????  6*
ksize
*
paddingSAME*
strides
2
stack0_enc4_last_pool/MaxPool?
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOpReadVariableOp>stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource*&
_output_shapes
:6Q*
dtype027
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?
&stack0_enc5_middle_expand_conv0/Conv2DConv2D&stack0_enc4_last_pool/MaxPool:output:0=stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2(
&stack0_enc5_middle_expand_conv0/Conv2D?
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOpReadVariableOp?stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype028
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?
'stack0_enc5_middle_expand_conv0/BiasAddBiasAdd/stack0_enc5_middle_expand_conv0/Conv2D:output:0>stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2)
'stack0_enc5_middle_expand_conv0/BiasAdd?
(stack0_enc5_middle_expand_act0_relu/ReluRelu0stack0_enc5_middle_expand_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q2*
(stack0_enc5_middle_expand_act0_relu/Relu?
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource*&
_output_shapes
:QQ*
dtype029
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
(stack0_enc6_middle_contract_conv0/Conv2DConv2D6stack0_enc5_middle_expand_act0_relu/Relu:activations:0?stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2*
(stack0_enc6_middle_contract_conv0/Conv2D?
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02:
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?
)stack0_enc6_middle_contract_conv0/BiasAddBiasAdd1stack0_enc6_middle_contract_conv0/Conv2D:output:0@stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2+
)stack0_enc6_middle_contract_conv0/BiasAdd?
*stack0_enc6_middle_contract_act0_relu/ReluRelu2stack0_enc6_middle_contract_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q2,
*stack0_enc6_middle_contract_act0_relu/Relu?
+stack0_dec0_s16_to_s8_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2-
+stack0_dec0_s16_to_s8_interp_bilinear/Const?
-stack0_dec0_s16_to_s8_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-stack0_dec0_s16_to_s8_interp_bilinear/Const_1?
)stack0_dec0_s16_to_s8_interp_bilinear/mulMul4stack0_dec0_s16_to_s8_interp_bilinear/Const:output:06stack0_dec0_s16_to_s8_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2+
)stack0_dec0_s16_to_s8_interp_bilinear/mul?
;stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinearResizeBilinear8stack0_enc6_middle_contract_act0_relu/Relu:activations:0-stack0_dec0_s16_to_s8_interp_bilinear/mul:z:0*
T0*/
_output_shapes
:?????????@@Q*
half_pixel_centers(2=
;stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear?
-stack0_dec0_s16_to_s8_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-stack0_dec0_s16_to_s8_skip_concat/concat/axis?
(stack0_dec0_s16_to_s8_skip_concat/concatConcatV2(stack0_enc3_act1_relu/Relu:activations:0Lstack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear:resized_images:06stack0_dec0_s16_to_s8_skip_concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?2*
(stack0_dec0_s16_to_s8_skip_concat/concat?
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOpReadVariableOpAstack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource*'
_output_shapes
:?6*
dtype02:
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?
)stack0_dec0_s16_to_s8_refine_conv0/Conv2DConv2D1stack0_dec0_s16_to_s8_skip_concat/concat:output:0@stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2+
)stack0_dec0_s16_to_s8_refine_conv0/Conv2D?
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpBstack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02;
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?
*stack0_dec0_s16_to_s8_refine_conv0/BiasAddBiasAdd2stack0_dec0_s16_to_s8_refine_conv0/Conv2D:output:0Astack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62,
*stack0_dec0_s16_to_s8_refine_conv0/BiasAdd?
0stack0_dec0_s16_to_s8_refine_conv0_act_relu/ReluRelu3stack0_dec0_s16_to_s8_refine_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@622
0stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu?
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOpReadVariableOpAstack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02:
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?
)stack0_dec0_s16_to_s8_refine_conv1/Conv2DConv2D>stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu:activations:0@stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2+
)stack0_dec0_s16_to_s8_refine_conv1/Conv2D?
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpBstack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02;
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?
*stack0_dec0_s16_to_s8_refine_conv1/BiasAddBiasAdd2stack0_dec0_s16_to_s8_refine_conv1/Conv2D:output:0Astack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62,
*stack0_dec0_s16_to_s8_refine_conv1/BiasAdd?
0stack0_dec0_s16_to_s8_refine_conv1_act_relu/ReluRelu3stack0_dec0_s16_to_s8_refine_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@622
0stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu?
*stack0_dec1_s8_to_s4_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   2,
*stack0_dec1_s8_to_s4_interp_bilinear/Const?
,stack0_dec1_s8_to_s4_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,stack0_dec1_s8_to_s4_interp_bilinear/Const_1?
(stack0_dec1_s8_to_s4_interp_bilinear/mulMul3stack0_dec1_s8_to_s4_interp_bilinear/Const:output:05stack0_dec1_s8_to_s4_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2*
(stack0_dec1_s8_to_s4_interp_bilinear/mul?
:stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinearResizeBilinear>stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu:activations:0,stack0_dec1_s8_to_s4_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????6*
half_pixel_centers(2<
:stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear?
,stack0_dec1_s8_to_s4_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,stack0_dec1_s8_to_s4_skip_concat/concat/axis?
'stack0_dec1_s8_to_s4_skip_concat/concatConcatV2(stack0_enc2_act1_relu/Relu:activations:0Kstack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear:resized_images:05stack0_dec1_s8_to_s4_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????Z2)
'stack0_dec1_s8_to_s4_skip_concat/concat?
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:Z$*
dtype029
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?
(stack0_dec1_s8_to_s4_refine_conv0/Conv2DConv2D0stack0_dec1_s8_to_s4_skip_concat/concat:output:0?stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2*
(stack0_dec1_s8_to_s4_refine_conv0/Conv2D?
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02:
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?
)stack0_dec1_s8_to_s4_refine_conv0/BiasAddBiasAdd1stack0_dec1_s8_to_s4_refine_conv0/Conv2D:output:0@stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2+
)stack0_dec1_s8_to_s4_refine_conv0/BiasAdd?
/stack0_dec1_s8_to_s4_refine_conv0_act_relu/ReluRelu2stack0_dec1_s8_to_s4_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$21
/stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu?
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOpReadVariableOp@stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype029
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?
(stack0_dec1_s8_to_s4_refine_conv1/Conv2DConv2D=stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu:activations:0?stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2*
(stack0_dec1_s8_to_s4_refine_conv1/Conv2D?
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02:
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?
)stack0_dec1_s8_to_s4_refine_conv1/BiasAddBiasAdd1stack0_dec1_s8_to_s4_refine_conv1/Conv2D:output:0@stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2+
)stack0_dec1_s8_to_s4_refine_conv1/BiasAdd?
/stack0_dec1_s8_to_s4_refine_conv1_act_relu/ReluRelu2stack0_dec1_s8_to_s4_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$21
/stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu?
*stack0_dec2_s4_to_s2_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2,
*stack0_dec2_s4_to_s2_interp_bilinear/Const?
,stack0_dec2_s4_to_s2_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,stack0_dec2_s4_to_s2_interp_bilinear/Const_1?
(stack0_dec2_s4_to_s2_interp_bilinear/mulMul3stack0_dec2_s4_to_s2_interp_bilinear/Const:output:05stack0_dec2_s4_to_s2_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2*
(stack0_dec2_s4_to_s2_interp_bilinear/mul?
:stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinearResizeBilinear=stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu:activations:0,stack0_dec2_s4_to_s2_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????$*
half_pixel_centers(2<
:stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear?
,stack0_dec2_s4_to_s2_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,stack0_dec2_s4_to_s2_skip_concat/concat/axis?
'stack0_dec2_s4_to_s2_skip_concat/concatConcatV2(stack0_enc1_act1_relu/Relu:activations:0Kstack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear:resized_images:05stack0_dec2_s4_to_s2_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????<2)
'stack0_dec2_s4_to_s2_skip_concat/concat?
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype029
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?
(stack0_dec2_s4_to_s2_refine_conv0/Conv2DConv2D0stack0_dec2_s4_to_s2_skip_concat/concat:output:0?stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(stack0_dec2_s4_to_s2_refine_conv0/Conv2D?
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?
)stack0_dec2_s4_to_s2_refine_conv0/BiasAddBiasAdd1stack0_dec2_s4_to_s2_refine_conv0/Conv2D:output:0@stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)stack0_dec2_s4_to_s2_refine_conv0/BiasAdd?
/stack0_dec2_s4_to_s2_refine_conv0_act_relu/ReluRelu2stack0_dec2_s4_to_s2_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????21
/stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu?
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOpReadVariableOp@stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?
(stack0_dec2_s4_to_s2_refine_conv1/Conv2DConv2D=stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu:activations:0?stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(stack0_dec2_s4_to_s2_refine_conv1/Conv2D?
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?
)stack0_dec2_s4_to_s2_refine_conv1/BiasAddBiasAdd1stack0_dec2_s4_to_s2_refine_conv1/Conv2D:output:0@stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)stack0_dec2_s4_to_s2_refine_conv1/BiasAdd?
/stack0_dec2_s4_to_s2_refine_conv1_act_relu/ReluRelu2stack0_dec2_s4_to_s2_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????21
/stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu?
*CentroidConfmapsHead/Conv2D/ReadVariableOpReadVariableOp3centroidconfmapshead_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*CentroidConfmapsHead/Conv2D/ReadVariableOp?
CentroidConfmapsHead/Conv2DConv2D=stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu:activations:02CentroidConfmapsHead/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
CentroidConfmapsHead/Conv2D?
+CentroidConfmapsHead/BiasAdd/ReadVariableOpReadVariableOp4centroidconfmapshead_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+CentroidConfmapsHead/BiasAdd/ReadVariableOp?
CentroidConfmapsHead/BiasAddBiasAdd$CentroidConfmapsHead/Conv2D:output:03CentroidConfmapsHead/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
CentroidConfmapsHead/BiasAdd?
IdentityIdentity%CentroidConfmapsHead/BiasAdd:output:0,^CentroidConfmapsHead/BiasAdd/ReadVariableOp+^CentroidConfmapsHead/Conv2D/ReadVariableOp:^stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp9^stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp:^stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp9^stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp9^stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp8^stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp9^stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp8^stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp9^stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp8^stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp9^stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp8^stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp)^stack0_enc0_conv0/BiasAdd/ReadVariableOp(^stack0_enc0_conv0/Conv2D/ReadVariableOp)^stack0_enc0_conv1/BiasAdd/ReadVariableOp(^stack0_enc0_conv1/Conv2D/ReadVariableOp)^stack0_enc1_conv0/BiasAdd/ReadVariableOp(^stack0_enc1_conv0/Conv2D/ReadVariableOp)^stack0_enc1_conv1/BiasAdd/ReadVariableOp(^stack0_enc1_conv1/Conv2D/ReadVariableOp)^stack0_enc2_conv0/BiasAdd/ReadVariableOp(^stack0_enc2_conv0/Conv2D/ReadVariableOp)^stack0_enc2_conv1/BiasAdd/ReadVariableOp(^stack0_enc2_conv1/Conv2D/ReadVariableOp)^stack0_enc3_conv0/BiasAdd/ReadVariableOp(^stack0_enc3_conv0/Conv2D/ReadVariableOp)^stack0_enc3_conv1/BiasAdd/ReadVariableOp(^stack0_enc3_conv1/Conv2D/ReadVariableOp7^stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp6^stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp9^stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp8^stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+CentroidConfmapsHead/BiasAdd/ReadVariableOp+CentroidConfmapsHead/BiasAdd/ReadVariableOp2X
*CentroidConfmapsHead/Conv2D/ReadVariableOp*CentroidConfmapsHead/Conv2D/ReadVariableOp2v
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp2t
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp2v
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp2t
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp2t
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp2r
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp2t
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp2r
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp2t
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp2r
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp2t
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp2r
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp2T
(stack0_enc0_conv0/BiasAdd/ReadVariableOp(stack0_enc0_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc0_conv0/Conv2D/ReadVariableOp'stack0_enc0_conv0/Conv2D/ReadVariableOp2T
(stack0_enc0_conv1/BiasAdd/ReadVariableOp(stack0_enc0_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc0_conv1/Conv2D/ReadVariableOp'stack0_enc0_conv1/Conv2D/ReadVariableOp2T
(stack0_enc1_conv0/BiasAdd/ReadVariableOp(stack0_enc1_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc1_conv0/Conv2D/ReadVariableOp'stack0_enc1_conv0/Conv2D/ReadVariableOp2T
(stack0_enc1_conv1/BiasAdd/ReadVariableOp(stack0_enc1_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc1_conv1/Conv2D/ReadVariableOp'stack0_enc1_conv1/Conv2D/ReadVariableOp2T
(stack0_enc2_conv0/BiasAdd/ReadVariableOp(stack0_enc2_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc2_conv0/Conv2D/ReadVariableOp'stack0_enc2_conv0/Conv2D/ReadVariableOp2T
(stack0_enc2_conv1/BiasAdd/ReadVariableOp(stack0_enc2_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc2_conv1/Conv2D/ReadVariableOp'stack0_enc2_conv1/Conv2D/ReadVariableOp2T
(stack0_enc3_conv0/BiasAdd/ReadVariableOp(stack0_enc3_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc3_conv0/Conv2D/ReadVariableOp'stack0_enc3_conv0/Conv2D/ReadVariableOp2T
(stack0_enc3_conv1/BiasAdd/ReadVariableOp(stack0_enc3_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc3_conv1/Conv2D/ReadVariableOp'stack0_enc3_conv1/Conv2D/ReadVariableOp2p
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp2n
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp2t
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp2r
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20938

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????$2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_20957

inputs8
conv2d_readvariableop_resource:$$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
܏
?"
G__inference_functional_1_layer_call_and_return_conditional_losses_20535

inputsJ
0stack0_enc0_conv0_conv2d_readvariableop_resource:?
1stack0_enc0_conv0_biasadd_readvariableop_resource:J
0stack0_enc0_conv1_conv2d_readvariableop_resource:?
1stack0_enc0_conv1_biasadd_readvariableop_resource:J
0stack0_enc1_conv0_conv2d_readvariableop_resource:?
1stack0_enc1_conv0_biasadd_readvariableop_resource:J
0stack0_enc1_conv1_conv2d_readvariableop_resource:?
1stack0_enc1_conv1_biasadd_readvariableop_resource:J
0stack0_enc2_conv0_conv2d_readvariableop_resource:$?
1stack0_enc2_conv0_biasadd_readvariableop_resource:$J
0stack0_enc2_conv1_conv2d_readvariableop_resource:$$?
1stack0_enc2_conv1_biasadd_readvariableop_resource:$J
0stack0_enc3_conv0_conv2d_readvariableop_resource:$6?
1stack0_enc3_conv0_biasadd_readvariableop_resource:6J
0stack0_enc3_conv1_conv2d_readvariableop_resource:66?
1stack0_enc3_conv1_biasadd_readvariableop_resource:6X
>stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource:6QM
?stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource:QZ
@stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource:QQO
Astack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource:Q\
Astack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource:?6P
Bstack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource:6[
Astack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource:66P
Bstack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource:6Z
@stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource:Z$O
Astack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource:$Z
@stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource:$$O
Astack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource:$Z
@stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource:<O
Astack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource:Z
@stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource:O
Astack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource:M
3centroidconfmapshead_conv2d_readvariableop_resource:B
4centroidconfmapshead_biasadd_readvariableop_resource:
identity??+CentroidConfmapsHead/BiasAdd/ReadVariableOp?*CentroidConfmapsHead/Conv2D/ReadVariableOp?9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?(stack0_enc0_conv0/BiasAdd/ReadVariableOp?'stack0_enc0_conv0/Conv2D/ReadVariableOp?(stack0_enc0_conv1/BiasAdd/ReadVariableOp?'stack0_enc0_conv1/Conv2D/ReadVariableOp?(stack0_enc1_conv0/BiasAdd/ReadVariableOp?'stack0_enc1_conv0/Conv2D/ReadVariableOp?(stack0_enc1_conv1/BiasAdd/ReadVariableOp?'stack0_enc1_conv1/Conv2D/ReadVariableOp?(stack0_enc2_conv0/BiasAdd/ReadVariableOp?'stack0_enc2_conv0/Conv2D/ReadVariableOp?(stack0_enc2_conv1/BiasAdd/ReadVariableOp?'stack0_enc2_conv1/Conv2D/ReadVariableOp?(stack0_enc3_conv0/BiasAdd/ReadVariableOp?'stack0_enc3_conv0/Conv2D/ReadVariableOp?(stack0_enc3_conv1/BiasAdd/ReadVariableOp?'stack0_enc3_conv1/Conv2D/ReadVariableOp?6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
'stack0_enc0_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc0_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc0_conv0/Conv2D/ReadVariableOp?
stack0_enc0_conv0/Conv2DConv2Dinputs/stack0_enc0_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc0_conv0/Conv2D?
(stack0_enc0_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc0_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc0_conv0/BiasAdd/ReadVariableOp?
stack0_enc0_conv0/BiasAddBiasAdd!stack0_enc0_conv0/Conv2D:output:00stack0_enc0_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_conv0/BiasAdd?
stack0_enc0_act0_relu/ReluRelu"stack0_enc0_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_act0_relu/Relu?
'stack0_enc0_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc0_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc0_conv1/Conv2D/ReadVariableOp?
stack0_enc0_conv1/Conv2DConv2D(stack0_enc0_act0_relu/Relu:activations:0/stack0_enc0_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc0_conv1/Conv2D?
(stack0_enc0_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc0_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc0_conv1/BiasAdd/ReadVariableOp?
stack0_enc0_conv1/BiasAddBiasAdd!stack0_enc0_conv1/Conv2D:output:00stack0_enc0_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_conv1/BiasAdd?
stack0_enc0_act1_relu/ReluRelu"stack0_enc0_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc0_act1_relu/Relu?
stack0_enc1_pool/MaxPoolMaxPool(stack0_enc0_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
stack0_enc1_pool/MaxPool?
'stack0_enc1_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc1_conv0_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc1_conv0/Conv2D/ReadVariableOp?
stack0_enc1_conv0/Conv2DConv2D!stack0_enc1_pool/MaxPool:output:0/stack0_enc1_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc1_conv0/Conv2D?
(stack0_enc1_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc1_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc1_conv0/BiasAdd/ReadVariableOp?
stack0_enc1_conv0/BiasAddBiasAdd!stack0_enc1_conv0/Conv2D:output:00stack0_enc1_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_conv0/BiasAdd?
stack0_enc1_act0_relu/ReluRelu"stack0_enc1_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_act0_relu/Relu?
'stack0_enc1_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc1_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02)
'stack0_enc1_conv1/Conv2D/ReadVariableOp?
stack0_enc1_conv1/Conv2DConv2D(stack0_enc1_act0_relu/Relu:activations:0/stack0_enc1_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
stack0_enc1_conv1/Conv2D?
(stack0_enc1_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc1_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02*
(stack0_enc1_conv1/BiasAdd/ReadVariableOp?
stack0_enc1_conv1/BiasAddBiasAdd!stack0_enc1_conv1/Conv2D:output:00stack0_enc1_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_conv1/BiasAdd?
stack0_enc1_act1_relu/ReluRelu"stack0_enc1_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????2
stack0_enc1_act1_relu/Relu?
stack0_enc2_pool/MaxPoolMaxPool(stack0_enc1_act1_relu/Relu:activations:0*1
_output_shapes
:???????????*
ksize
*
paddingSAME*
strides
2
stack0_enc2_pool/MaxPool?
'stack0_enc2_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc2_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$*
dtype02)
'stack0_enc2_conv0/Conv2D/ReadVariableOp?
stack0_enc2_conv0/Conv2DConv2D!stack0_enc2_pool/MaxPool:output:0/stack0_enc2_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
stack0_enc2_conv0/Conv2D?
(stack0_enc2_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc2_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02*
(stack0_enc2_conv0/BiasAdd/ReadVariableOp?
stack0_enc2_conv0/BiasAddBiasAdd!stack0_enc2_conv0/Conv2D:output:00stack0_enc2_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_conv0/BiasAdd?
stack0_enc2_act0_relu/ReluRelu"stack0_enc2_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_act0_relu/Relu?
'stack0_enc2_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc2_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02)
'stack0_enc2_conv1/Conv2D/ReadVariableOp?
stack0_enc2_conv1/Conv2DConv2D(stack0_enc2_act0_relu/Relu:activations:0/stack0_enc2_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
stack0_enc2_conv1/Conv2D?
(stack0_enc2_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc2_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02*
(stack0_enc2_conv1/BiasAdd/ReadVariableOp?
stack0_enc2_conv1/BiasAddBiasAdd!stack0_enc2_conv1/Conv2D:output:00stack0_enc2_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_conv1/BiasAdd?
stack0_enc2_act1_relu/ReluRelu"stack0_enc2_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$2
stack0_enc2_act1_relu/Relu?
stack0_enc3_pool/MaxPoolMaxPool(stack0_enc2_act1_relu/Relu:activations:0*/
_output_shapes
:?????????@@$*
ksize
*
paddingSAME*
strides
2
stack0_enc3_pool/MaxPool?
'stack0_enc3_conv0/Conv2D/ReadVariableOpReadVariableOp0stack0_enc3_conv0_conv2d_readvariableop_resource*&
_output_shapes
:$6*
dtype02)
'stack0_enc3_conv0/Conv2D/ReadVariableOp?
stack0_enc3_conv0/Conv2DConv2D!stack0_enc3_pool/MaxPool:output:0/stack0_enc3_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
stack0_enc3_conv0/Conv2D?
(stack0_enc3_conv0/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc3_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02*
(stack0_enc3_conv0/BiasAdd/ReadVariableOp?
stack0_enc3_conv0/BiasAddBiasAdd!stack0_enc3_conv0/Conv2D:output:00stack0_enc3_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_conv0/BiasAdd?
stack0_enc3_act0_relu/ReluRelu"stack0_enc3_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_act0_relu/Relu?
'stack0_enc3_conv1/Conv2D/ReadVariableOpReadVariableOp0stack0_enc3_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02)
'stack0_enc3_conv1/Conv2D/ReadVariableOp?
stack0_enc3_conv1/Conv2DConv2D(stack0_enc3_act0_relu/Relu:activations:0/stack0_enc3_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2
stack0_enc3_conv1/Conv2D?
(stack0_enc3_conv1/BiasAdd/ReadVariableOpReadVariableOp1stack0_enc3_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02*
(stack0_enc3_conv1/BiasAdd/ReadVariableOp?
stack0_enc3_conv1/BiasAddBiasAdd!stack0_enc3_conv1/Conv2D:output:00stack0_enc3_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_conv1/BiasAdd?
stack0_enc3_act1_relu/ReluRelu"stack0_enc3_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@62
stack0_enc3_act1_relu/Relu?
stack0_enc4_last_pool/MaxPoolMaxPool(stack0_enc3_act1_relu/Relu:activations:0*/
_output_shapes
:?????????  6*
ksize
*
paddingSAME*
strides
2
stack0_enc4_last_pool/MaxPool?
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOpReadVariableOp>stack0_enc5_middle_expand_conv0_conv2d_readvariableop_resource*&
_output_shapes
:6Q*
dtype027
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp?
&stack0_enc5_middle_expand_conv0/Conv2DConv2D&stack0_enc4_last_pool/MaxPool:output:0=stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2(
&stack0_enc5_middle_expand_conv0/Conv2D?
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOpReadVariableOp?stack0_enc5_middle_expand_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype028
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp?
'stack0_enc5_middle_expand_conv0/BiasAddBiasAdd/stack0_enc5_middle_expand_conv0/Conv2D:output:0>stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2)
'stack0_enc5_middle_expand_conv0/BiasAdd?
(stack0_enc5_middle_expand_act0_relu/ReluRelu0stack0_enc5_middle_expand_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q2*
(stack0_enc5_middle_expand_act0_relu/Relu?
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_enc6_middle_contract_conv0_conv2d_readvariableop_resource*&
_output_shapes
:QQ*
dtype029
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp?
(stack0_enc6_middle_contract_conv0/Conv2DConv2D6stack0_enc5_middle_expand_act0_relu/Relu:activations:0?stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2*
(stack0_enc6_middle_contract_conv0/Conv2D?
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_enc6_middle_contract_conv0_biasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02:
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp?
)stack0_enc6_middle_contract_conv0/BiasAddBiasAdd1stack0_enc6_middle_contract_conv0/Conv2D:output:0@stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2+
)stack0_enc6_middle_contract_conv0/BiasAdd?
*stack0_enc6_middle_contract_act0_relu/ReluRelu2stack0_enc6_middle_contract_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????  Q2,
*stack0_enc6_middle_contract_act0_relu/Relu?
+stack0_dec0_s16_to_s8_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"        2-
+stack0_dec0_s16_to_s8_interp_bilinear/Const?
-stack0_dec0_s16_to_s8_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2/
-stack0_dec0_s16_to_s8_interp_bilinear/Const_1?
)stack0_dec0_s16_to_s8_interp_bilinear/mulMul4stack0_dec0_s16_to_s8_interp_bilinear/Const:output:06stack0_dec0_s16_to_s8_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2+
)stack0_dec0_s16_to_s8_interp_bilinear/mul?
;stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinearResizeBilinear8stack0_enc6_middle_contract_act0_relu/Relu:activations:0-stack0_dec0_s16_to_s8_interp_bilinear/mul:z:0*
T0*/
_output_shapes
:?????????@@Q*
half_pixel_centers(2=
;stack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear?
-stack0_dec0_s16_to_s8_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2/
-stack0_dec0_s16_to_s8_skip_concat/concat/axis?
(stack0_dec0_s16_to_s8_skip_concat/concatConcatV2(stack0_enc3_act1_relu/Relu:activations:0Lstack0_dec0_s16_to_s8_interp_bilinear/resize/ResizeBilinear:resized_images:06stack0_dec0_s16_to_s8_skip_concat/concat/axis:output:0*
N*
T0*0
_output_shapes
:?????????@@?2*
(stack0_dec0_s16_to_s8_skip_concat/concat?
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOpReadVariableOpAstack0_dec0_s16_to_s8_refine_conv0_conv2d_readvariableop_resource*'
_output_shapes
:?6*
dtype02:
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp?
)stack0_dec0_s16_to_s8_refine_conv0/Conv2DConv2D1stack0_dec0_s16_to_s8_skip_concat/concat:output:0@stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2+
)stack0_dec0_s16_to_s8_refine_conv0/Conv2D?
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpBstack0_dec0_s16_to_s8_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02;
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp?
*stack0_dec0_s16_to_s8_refine_conv0/BiasAddBiasAdd2stack0_dec0_s16_to_s8_refine_conv0/Conv2D:output:0Astack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62,
*stack0_dec0_s16_to_s8_refine_conv0/BiasAdd?
0stack0_dec0_s16_to_s8_refine_conv0_act_relu/ReluRelu3stack0_dec0_s16_to_s8_refine_conv0/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@622
0stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu?
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOpReadVariableOpAstack0_dec0_s16_to_s8_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:66*
dtype02:
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp?
)stack0_dec0_s16_to_s8_refine_conv1/Conv2DConv2D>stack0_dec0_s16_to_s8_refine_conv0_act_relu/Relu:activations:0@stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@6*
paddingSAME*
strides
2+
)stack0_dec0_s16_to_s8_refine_conv1/Conv2D?
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpBstack0_dec0_s16_to_s8_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:6*
dtype02;
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp?
*stack0_dec0_s16_to_s8_refine_conv1/BiasAddBiasAdd2stack0_dec0_s16_to_s8_refine_conv1/Conv2D:output:0Astack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@@62,
*stack0_dec0_s16_to_s8_refine_conv1/BiasAdd?
0stack0_dec0_s16_to_s8_refine_conv1_act_relu/ReluRelu3stack0_dec0_s16_to_s8_refine_conv1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@@622
0stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu?
*stack0_dec1_s8_to_s4_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"@   @   2,
*stack0_dec1_s8_to_s4_interp_bilinear/Const?
,stack0_dec1_s8_to_s4_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,stack0_dec1_s8_to_s4_interp_bilinear/Const_1?
(stack0_dec1_s8_to_s4_interp_bilinear/mulMul3stack0_dec1_s8_to_s4_interp_bilinear/Const:output:05stack0_dec1_s8_to_s4_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2*
(stack0_dec1_s8_to_s4_interp_bilinear/mul?
:stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinearResizeBilinear>stack0_dec0_s16_to_s8_refine_conv1_act_relu/Relu:activations:0,stack0_dec1_s8_to_s4_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????6*
half_pixel_centers(2<
:stack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear?
,stack0_dec1_s8_to_s4_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,stack0_dec1_s8_to_s4_skip_concat/concat/axis?
'stack0_dec1_s8_to_s4_skip_concat/concatConcatV2(stack0_enc2_act1_relu/Relu:activations:0Kstack0_dec1_s8_to_s4_interp_bilinear/resize/ResizeBilinear:resized_images:05stack0_dec1_s8_to_s4_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????Z2)
'stack0_dec1_s8_to_s4_skip_concat/concat?
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_dec1_s8_to_s4_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:Z$*
dtype029
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp?
(stack0_dec1_s8_to_s4_refine_conv0/Conv2DConv2D0stack0_dec1_s8_to_s4_skip_concat/concat:output:0?stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2*
(stack0_dec1_s8_to_s4_refine_conv0/Conv2D?
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec1_s8_to_s4_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02:
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp?
)stack0_dec1_s8_to_s4_refine_conv0/BiasAddBiasAdd1stack0_dec1_s8_to_s4_refine_conv0/Conv2D:output:0@stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2+
)stack0_dec1_s8_to_s4_refine_conv0/BiasAdd?
/stack0_dec1_s8_to_s4_refine_conv0_act_relu/ReluRelu2stack0_dec1_s8_to_s4_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$21
/stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu?
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOpReadVariableOp@stack0_dec1_s8_to_s4_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype029
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp?
(stack0_dec1_s8_to_s4_refine_conv1/Conv2DConv2D=stack0_dec1_s8_to_s4_refine_conv0_act_relu/Relu:activations:0?stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2*
(stack0_dec1_s8_to_s4_refine_conv1/Conv2D?
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec1_s8_to_s4_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02:
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp?
)stack0_dec1_s8_to_s4_refine_conv1/BiasAddBiasAdd1stack0_dec1_s8_to_s4_refine_conv1/Conv2D:output:0@stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2+
)stack0_dec1_s8_to_s4_refine_conv1/BiasAdd?
/stack0_dec1_s8_to_s4_refine_conv1_act_relu/ReluRelu2stack0_dec1_s8_to_s4_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????$21
/stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu?
*stack0_dec2_s4_to_s2_interp_bilinear/ConstConst*
_output_shapes
:*
dtype0*
valueB"?   ?   2,
*stack0_dec2_s4_to_s2_interp_bilinear/Const?
,stack0_dec2_s4_to_s2_interp_bilinear/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2.
,stack0_dec2_s4_to_s2_interp_bilinear/Const_1?
(stack0_dec2_s4_to_s2_interp_bilinear/mulMul3stack0_dec2_s4_to_s2_interp_bilinear/Const:output:05stack0_dec2_s4_to_s2_interp_bilinear/Const_1:output:0*
T0*
_output_shapes
:2*
(stack0_dec2_s4_to_s2_interp_bilinear/mul?
:stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinearResizeBilinear=stack0_dec1_s8_to_s4_refine_conv1_act_relu/Relu:activations:0,stack0_dec2_s4_to_s2_interp_bilinear/mul:z:0*
T0*1
_output_shapes
:???????????$*
half_pixel_centers(2<
:stack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear?
,stack0_dec2_s4_to_s2_skip_concat/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2.
,stack0_dec2_s4_to_s2_skip_concat/concat/axis?
'stack0_dec2_s4_to_s2_skip_concat/concatConcatV2(stack0_enc1_act1_relu/Relu:activations:0Kstack0_dec2_s4_to_s2_interp_bilinear/resize/ResizeBilinear:resized_images:05stack0_dec2_s4_to_s2_skip_concat/concat/axis:output:0*
N*
T0*1
_output_shapes
:???????????<2)
'stack0_dec2_s4_to_s2_skip_concat/concat?
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOpReadVariableOp@stack0_dec2_s4_to_s2_refine_conv0_conv2d_readvariableop_resource*&
_output_shapes
:<*
dtype029
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp?
(stack0_dec2_s4_to_s2_refine_conv0/Conv2DConv2D0stack0_dec2_s4_to_s2_skip_concat/concat:output:0?stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(stack0_dec2_s4_to_s2_refine_conv0/Conv2D?
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec2_s4_to_s2_refine_conv0_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp?
)stack0_dec2_s4_to_s2_refine_conv0/BiasAddBiasAdd1stack0_dec2_s4_to_s2_refine_conv0/Conv2D:output:0@stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)stack0_dec2_s4_to_s2_refine_conv0/BiasAdd?
/stack0_dec2_s4_to_s2_refine_conv0_act_relu/ReluRelu2stack0_dec2_s4_to_s2_refine_conv0/BiasAdd:output:0*
T0*1
_output_shapes
:???????????21
/stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu?
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOpReadVariableOp@stack0_dec2_s4_to_s2_refine_conv1_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype029
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp?
(stack0_dec2_s4_to_s2_refine_conv1/Conv2DConv2D=stack0_dec2_s4_to_s2_refine_conv0_act_relu/Relu:activations:0?stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2*
(stack0_dec2_s4_to_s2_refine_conv1/Conv2D?
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOpReadVariableOpAstack0_dec2_s4_to_s2_refine_conv1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp?
)stack0_dec2_s4_to_s2_refine_conv1/BiasAddBiasAdd1stack0_dec2_s4_to_s2_refine_conv1/Conv2D:output:0@stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2+
)stack0_dec2_s4_to_s2_refine_conv1/BiasAdd?
/stack0_dec2_s4_to_s2_refine_conv1_act_relu/ReluRelu2stack0_dec2_s4_to_s2_refine_conv1/BiasAdd:output:0*
T0*1
_output_shapes
:???????????21
/stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu?
*CentroidConfmapsHead/Conv2D/ReadVariableOpReadVariableOp3centroidconfmapshead_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02,
*CentroidConfmapsHead/Conv2D/ReadVariableOp?
CentroidConfmapsHead/Conv2DConv2D=stack0_dec2_s4_to_s2_refine_conv1_act_relu/Relu:activations:02CentroidConfmapsHead/Conv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
CentroidConfmapsHead/Conv2D?
+CentroidConfmapsHead/BiasAdd/ReadVariableOpReadVariableOp4centroidconfmapshead_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+CentroidConfmapsHead/BiasAdd/ReadVariableOp?
CentroidConfmapsHead/BiasAddBiasAdd$CentroidConfmapsHead/Conv2D:output:03CentroidConfmapsHead/BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2
CentroidConfmapsHead/BiasAdd?
IdentityIdentity%CentroidConfmapsHead/BiasAdd:output:0,^CentroidConfmapsHead/BiasAdd/ReadVariableOp+^CentroidConfmapsHead/Conv2D/ReadVariableOp:^stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp9^stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp:^stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp9^stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp9^stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp8^stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp9^stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp8^stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp9^stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp8^stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp9^stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp8^stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp)^stack0_enc0_conv0/BiasAdd/ReadVariableOp(^stack0_enc0_conv0/Conv2D/ReadVariableOp)^stack0_enc0_conv1/BiasAdd/ReadVariableOp(^stack0_enc0_conv1/Conv2D/ReadVariableOp)^stack0_enc1_conv0/BiasAdd/ReadVariableOp(^stack0_enc1_conv0/Conv2D/ReadVariableOp)^stack0_enc1_conv1/BiasAdd/ReadVariableOp(^stack0_enc1_conv1/Conv2D/ReadVariableOp)^stack0_enc2_conv0/BiasAdd/ReadVariableOp(^stack0_enc2_conv0/Conv2D/ReadVariableOp)^stack0_enc2_conv1/BiasAdd/ReadVariableOp(^stack0_enc2_conv1/Conv2D/ReadVariableOp)^stack0_enc3_conv0/BiasAdd/ReadVariableOp(^stack0_enc3_conv0/Conv2D/ReadVariableOp)^stack0_enc3_conv1/BiasAdd/ReadVariableOp(^stack0_enc3_conv1/Conv2D/ReadVariableOp7^stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp6^stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp9^stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp8^stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+CentroidConfmapsHead/BiasAdd/ReadVariableOp+CentroidConfmapsHead/BiasAdd/ReadVariableOp2X
*CentroidConfmapsHead/Conv2D/ReadVariableOp*CentroidConfmapsHead/Conv2D/ReadVariableOp2v
9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp9stack0_dec0_s16_to_s8_refine_conv0/BiasAdd/ReadVariableOp2t
8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp8stack0_dec0_s16_to_s8_refine_conv0/Conv2D/ReadVariableOp2v
9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp9stack0_dec0_s16_to_s8_refine_conv1/BiasAdd/ReadVariableOp2t
8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp8stack0_dec0_s16_to_s8_refine_conv1/Conv2D/ReadVariableOp2t
8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp8stack0_dec1_s8_to_s4_refine_conv0/BiasAdd/ReadVariableOp2r
7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp7stack0_dec1_s8_to_s4_refine_conv0/Conv2D/ReadVariableOp2t
8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp8stack0_dec1_s8_to_s4_refine_conv1/BiasAdd/ReadVariableOp2r
7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp7stack0_dec1_s8_to_s4_refine_conv1/Conv2D/ReadVariableOp2t
8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp8stack0_dec2_s4_to_s2_refine_conv0/BiasAdd/ReadVariableOp2r
7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp7stack0_dec2_s4_to_s2_refine_conv0/Conv2D/ReadVariableOp2t
8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp8stack0_dec2_s4_to_s2_refine_conv1/BiasAdd/ReadVariableOp2r
7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp7stack0_dec2_s4_to_s2_refine_conv1/Conv2D/ReadVariableOp2T
(stack0_enc0_conv0/BiasAdd/ReadVariableOp(stack0_enc0_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc0_conv0/Conv2D/ReadVariableOp'stack0_enc0_conv0/Conv2D/ReadVariableOp2T
(stack0_enc0_conv1/BiasAdd/ReadVariableOp(stack0_enc0_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc0_conv1/Conv2D/ReadVariableOp'stack0_enc0_conv1/Conv2D/ReadVariableOp2T
(stack0_enc1_conv0/BiasAdd/ReadVariableOp(stack0_enc1_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc1_conv0/Conv2D/ReadVariableOp'stack0_enc1_conv0/Conv2D/ReadVariableOp2T
(stack0_enc1_conv1/BiasAdd/ReadVariableOp(stack0_enc1_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc1_conv1/Conv2D/ReadVariableOp'stack0_enc1_conv1/Conv2D/ReadVariableOp2T
(stack0_enc2_conv0/BiasAdd/ReadVariableOp(stack0_enc2_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc2_conv0/Conv2D/ReadVariableOp'stack0_enc2_conv0/Conv2D/ReadVariableOp2T
(stack0_enc2_conv1/BiasAdd/ReadVariableOp(stack0_enc2_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc2_conv1/Conv2D/ReadVariableOp'stack0_enc2_conv1/Conv2D/ReadVariableOp2T
(stack0_enc3_conv0/BiasAdd/ReadVariableOp(stack0_enc3_conv0/BiasAdd/ReadVariableOp2R
'stack0_enc3_conv0/Conv2D/ReadVariableOp'stack0_enc3_conv0/Conv2D/ReadVariableOp2T
(stack0_enc3_conv1/BiasAdd/ReadVariableOp(stack0_enc3_conv1/BiasAdd/ReadVariableOp2R
'stack0_enc3_conv1/Conv2D/ReadVariableOp'stack0_enc3_conv1/Conv2D/ReadVariableOp2p
6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp6stack0_enc5_middle_expand_conv0/BiasAdd/ReadVariableOp2n
5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp5stack0_enc5_middle_expand_conv0/Conv2D/ReadVariableOp2t
8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp8stack0_enc6_middle_contract_conv0/BiasAdd/ReadVariableOp2r
7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp7stack0_enc6_middle_contract_conv0/Conv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc0_conv1_layer_call_fn_20573

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_187182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_20815

inputs8
conv2d_readvariableop_resource:QQ-
biasadd_readvariableop_resource:Q
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:QQ*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  Q: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  Q
 
_user_specified_nameinputs
?
f
J__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_fn_20962

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190292
PartitionedCallv
IdentityIdentityPartitionedCall:output:0*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????$:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
??
?
G__inference_functional_1_layer_call_and_return_conditional_losses_19104

inputs1
stack0_enc0_conv0_18696:%
stack0_enc0_conv0_18698:1
stack0_enc0_conv1_18719:%
stack0_enc0_conv1_18721:1
stack0_enc1_conv0_18743:%
stack0_enc1_conv0_18745:1
stack0_enc1_conv1_18766:%
stack0_enc1_conv1_18768:1
stack0_enc2_conv0_18790:$%
stack0_enc2_conv0_18792:$1
stack0_enc2_conv1_18813:$$%
stack0_enc2_conv1_18815:$1
stack0_enc3_conv0_18837:$6%
stack0_enc3_conv0_18839:61
stack0_enc3_conv1_18860:66%
stack0_enc3_conv1_18862:6?
%stack0_enc5_middle_expand_conv0_18884:6Q3
%stack0_enc5_middle_expand_conv0_18886:QA
'stack0_enc6_middle_contract_conv0_18907:QQ5
'stack0_enc6_middle_contract_conv0_18909:QC
(stack0_dec0_s16_to_s8_refine_conv0_18940:?66
(stack0_dec0_s16_to_s8_refine_conv0_18942:6B
(stack0_dec0_s16_to_s8_refine_conv1_18963:666
(stack0_dec0_s16_to_s8_refine_conv1_18965:6A
'stack0_dec1_s8_to_s4_refine_conv0_18996:Z$5
'stack0_dec1_s8_to_s4_refine_conv0_18998:$A
'stack0_dec1_s8_to_s4_refine_conv1_19019:$$5
'stack0_dec1_s8_to_s4_refine_conv1_19021:$A
'stack0_dec2_s4_to_s2_refine_conv0_19052:<5
'stack0_dec2_s4_to_s2_refine_conv0_19054:A
'stack0_dec2_s4_to_s2_refine_conv1_19075:5
'stack0_dec2_s4_to_s2_refine_conv1_19077:4
centroidconfmapshead_19098:(
centroidconfmapshead_19100:
identity??,CentroidConfmapsHead/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?)stack0_enc0_conv0/StatefulPartitionedCall?)stack0_enc0_conv1/StatefulPartitionedCall?)stack0_enc1_conv0/StatefulPartitionedCall?)stack0_enc1_conv1/StatefulPartitionedCall?)stack0_enc2_conv0/StatefulPartitionedCall?)stack0_enc2_conv1/StatefulPartitionedCall?)stack0_enc3_conv0/StatefulPartitionedCall?)stack0_enc3_conv1/StatefulPartitionedCall?7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
)stack0_enc0_conv0/StatefulPartitionedCallStatefulPartitionedCallinputsstack0_enc0_conv0_18696stack0_enc0_conv0_18698*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_186952+
)stack0_enc0_conv0/StatefulPartitionedCall?
%stack0_enc0_act0_relu/PartitionedCallPartitionedCall2stack0_enc0_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_187062'
%stack0_enc0_act0_relu/PartitionedCall?
)stack0_enc0_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc0_act0_relu/PartitionedCall:output:0stack0_enc0_conv1_18719stack0_enc0_conv1_18721*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_187182+
)stack0_enc0_conv1/StatefulPartitionedCall?
%stack0_enc0_act1_relu/PartitionedCallPartitionedCall2stack0_enc0_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_187292'
%stack0_enc0_act1_relu/PartitionedCall?
 stack0_enc1_pool/PartitionedCallPartitionedCall.stack0_enc0_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_185792"
 stack0_enc1_pool/PartitionedCall?
)stack0_enc1_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc1_pool/PartitionedCall:output:0stack0_enc1_conv0_18743stack0_enc1_conv0_18745*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_187422+
)stack0_enc1_conv0/StatefulPartitionedCall?
%stack0_enc1_act0_relu/PartitionedCallPartitionedCall2stack0_enc1_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_187532'
%stack0_enc1_act0_relu/PartitionedCall?
)stack0_enc1_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc1_act0_relu/PartitionedCall:output:0stack0_enc1_conv1_18766stack0_enc1_conv1_18768*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_187652+
)stack0_enc1_conv1/StatefulPartitionedCall?
%stack0_enc1_act1_relu/PartitionedCallPartitionedCall2stack0_enc1_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_187762'
%stack0_enc1_act1_relu/PartitionedCall?
 stack0_enc2_pool/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_185912"
 stack0_enc2_pool/PartitionedCall?
)stack0_enc2_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc2_pool/PartitionedCall:output:0stack0_enc2_conv0_18790stack0_enc2_conv0_18792*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_187892+
)stack0_enc2_conv0/StatefulPartitionedCall?
%stack0_enc2_act0_relu/PartitionedCallPartitionedCall2stack0_enc2_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_188002'
%stack0_enc2_act0_relu/PartitionedCall?
)stack0_enc2_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc2_act0_relu/PartitionedCall:output:0stack0_enc2_conv1_18813stack0_enc2_conv1_18815*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_188122+
)stack0_enc2_conv1/StatefulPartitionedCall?
%stack0_enc2_act1_relu/PartitionedCallPartitionedCall2stack0_enc2_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_188232'
%stack0_enc2_act1_relu/PartitionedCall?
 stack0_enc3_pool/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *T
fORM
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_186032"
 stack0_enc3_pool/PartitionedCall?
)stack0_enc3_conv0/StatefulPartitionedCallStatefulPartitionedCall)stack0_enc3_pool/PartitionedCall:output:0stack0_enc3_conv0_18837stack0_enc3_conv0_18839*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_188362+
)stack0_enc3_conv0/StatefulPartitionedCall?
%stack0_enc3_act0_relu/PartitionedCallPartitionedCall2stack0_enc3_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_188472'
%stack0_enc3_act0_relu/PartitionedCall?
)stack0_enc3_conv1/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc3_act0_relu/PartitionedCall:output:0stack0_enc3_conv1_18860stack0_enc3_conv1_18862*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_188592+
)stack0_enc3_conv1/StatefulPartitionedCall?
%stack0_enc3_act1_relu/PartitionedCallPartitionedCall2stack0_enc3_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_188702'
%stack0_enc3_act1_relu/PartitionedCall?
%stack0_enc4_last_pool/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_186152'
%stack0_enc4_last_pool/PartitionedCall?
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCallStatefulPartitionedCall.stack0_enc4_last_pool/PartitionedCall:output:0%stack0_enc5_middle_expand_conv0_18884%stack0_enc5_middle_expand_conv0_18886*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *c
f^R\
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_1888329
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall?
3stack0_enc5_middle_expand_act0_relu/PartitionedCallPartitionedCall@stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *g
fbR`
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_1889425
3stack0_enc5_middle_expand_act0_relu/PartitionedCall?
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCallStatefulPartitionedCall<stack0_enc5_middle_expand_act0_relu/PartitionedCall:output:0'stack0_enc6_middle_contract_conv0_18907'stack0_enc6_middle_contract_conv0_18909*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_189062;
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall?
5stack0_enc6_middle_contract_act0_relu/PartitionedCallPartitionedCallBstack0_enc6_middle_contract_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????  Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_1891727
5stack0_enc6_middle_contract_act0_relu/PartitionedCall?
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCallPartitionedCall>stack0_enc6_middle_contract_act0_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????Q* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *i
fdRb
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_1863427
5stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall?
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCallPartitionedCall.stack0_enc3_act1_relu/PartitionedCall:output:0>stack0_dec0_s16_to_s8_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@@?* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_1892723
1stack0_dec0_s16_to_s8_skip_concat/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall:stack0_dec0_s16_to_s8_skip_concat/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv0_18940(stack0_dec0_s16_to_s8_refine_conv0_18942*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_189392<
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_189502=
;stack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall?
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallDstack0_dec0_s16_to_s8_refine_conv0_act_relu/PartitionedCall:output:0(stack0_dec0_s16_to_s8_refine_conv1_18963(stack0_dec0_s16_to_s8_refine_conv1_18965*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_189622<
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall?
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCallPartitionedCallCstack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *o
fjRh
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_189732=
;stack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall?
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCallPartitionedCallDstack0_dec0_s16_to_s8_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????6* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_1865326
4stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall?
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCallPartitionedCall.stack0_enc2_act1_relu/PartitionedCall:output:0=stack0_dec1_s8_to_s4_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????Z* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_1898322
0stack0_dec1_s8_to_s4_skip_concat/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec1_s8_to_s4_skip_concat/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv0_18996'stack0_dec1_s8_to_s4_refine_conv0_18998*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_189952;
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190062<
:stack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall?
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec1_s8_to_s4_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec1_s8_to_s4_refine_conv1_19019'stack0_dec1_s8_to_s4_refine_conv1_19021*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_190182;
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall?
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190292<
:stack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall?
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCallPartitionedCallCstack0_dec1_s8_to_s4_refine_conv1_act_relu/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????$* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *h
fcRa
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_1867226
4stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall?
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCallPartitionedCall.stack0_enc1_act1_relu/PartitionedCall:output:0=stack0_dec2_s4_to_s2_interp_bilinear/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????<* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *d
f_R]
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_1903922
0stack0_dec2_s4_to_s2_skip_concat/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCallStatefulPartitionedCall9stack0_dec2_s4_to_s2_skip_concat/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv0_19052'stack0_dec2_s4_to_s2_refine_conv0_19054*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_190512;
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_190622<
:stack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall?
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv0_act_relu/PartitionedCall:output:0'stack0_dec2_s4_to_s2_refine_conv1_19075'stack0_dec2_s4_to_s2_refine_conv1_19077*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *e
f`R^
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_190742;
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall?
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCallPartitionedCallBstack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *n
fiRg
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_190852<
:stack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall?
,CentroidConfmapsHead/StatefulPartitionedCallStatefulPartitionedCallCstack0_dec2_s4_to_s2_refine_conv1_act_relu/PartitionedCall:output:0centroidconfmapshead_19098centroidconfmapshead_19100*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_190972.
,CentroidConfmapsHead/StatefulPartitionedCall?
IdentityIdentity5CentroidConfmapsHead/StatefulPartitionedCall:output:0-^CentroidConfmapsHead/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall;^stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall:^stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall:^stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall*^stack0_enc0_conv0/StatefulPartitionedCall*^stack0_enc0_conv1/StatefulPartitionedCall*^stack0_enc1_conv0/StatefulPartitionedCall*^stack0_enc1_conv1/StatefulPartitionedCall*^stack0_enc2_conv0/StatefulPartitionedCall*^stack0_enc2_conv1/StatefulPartitionedCall*^stack0_enc3_conv0/StatefulPartitionedCall*^stack0_enc3_conv1/StatefulPartitionedCall8^stack0_enc5_middle_expand_conv0/StatefulPartitionedCall:^stack0_enc6_middle_contract_conv0/StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2\
,CentroidConfmapsHead/StatefulPartitionedCall,CentroidConfmapsHead/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv0/StatefulPartitionedCall2x
:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall:stack0_dec0_s16_to_s8_refine_conv1/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv0/StatefulPartitionedCall2v
9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall9stack0_dec1_s8_to_s4_refine_conv1/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv0/StatefulPartitionedCall2v
9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall9stack0_dec2_s4_to_s2_refine_conv1/StatefulPartitionedCall2V
)stack0_enc0_conv0/StatefulPartitionedCall)stack0_enc0_conv0/StatefulPartitionedCall2V
)stack0_enc0_conv1/StatefulPartitionedCall)stack0_enc0_conv1/StatefulPartitionedCall2V
)stack0_enc1_conv0/StatefulPartitionedCall)stack0_enc1_conv0/StatefulPartitionedCall2V
)stack0_enc1_conv1/StatefulPartitionedCall)stack0_enc1_conv1/StatefulPartitionedCall2V
)stack0_enc2_conv0/StatefulPartitionedCall)stack0_enc2_conv0/StatefulPartitionedCall2V
)stack0_enc2_conv1/StatefulPartitionedCall)stack0_enc2_conv1/StatefulPartitionedCall2V
)stack0_enc3_conv0/StatefulPartitionedCall)stack0_enc3_conv0/StatefulPartitionedCall2V
)stack0_enc3_conv1/StatefulPartitionedCall)stack0_enc3_conv1/StatefulPartitionedCall2r
7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall7stack0_enc5_middle_expand_conv0/StatefulPartitionedCall2v
9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall9stack0_enc6_middle_contract_conv0/StatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_18615

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
4__inference_CentroidConfmapsHead_layer_call_fn_21047

inputs!
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_190972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?

?
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_19018

inputs8
conv2d_readvariableop_resource:$$-
biasadd_readvariableop_resource:$
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:$$*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????$2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????$2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????$: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????$
 
_user_specified_nameinputs
?
Q
5__inference_stack0_enc4_last_pool_layer_call_fn_18621

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_186152
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_19097

inputs8
conv2d_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*1
_output_shapes
:???????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:???????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
?
1__inference_stack0_enc3_conv0_layer_call_fn_20718

inputs!
unknown:$6
	unknown_0:6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@@6*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *U
fPRN
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_188362
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@@62

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@@$: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@@$
 
_user_specified_nameinputs
?

?
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_18883

inputs8
conv2d_readvariableop_resource:6Q-
biasadd_readvariableop_resource:Q
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:6Q*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:Q*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????  Q2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????  Q2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  6: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  6
 
_user_specified_nameinputs
?
l
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_20593

inputs
identityX
ReluReluinputs*
T0*1
_output_shapes
:???????????2
Relup
IdentityIdentityRelu:activations:0*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:???????????:Y U
1
_output_shapes
:???????????
 
_user_specified_nameinputs
?
g
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_18603

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingSAME*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?	
,__inference_functional_1_layer_call_fn_19175	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:$
	unknown_8:$#
	unknown_9:$$

unknown_10:$$

unknown_11:$6

unknown_12:6$

unknown_13:66

unknown_14:6$

unknown_15:6Q

unknown_16:Q$

unknown_17:QQ

unknown_18:Q%

unknown_19:?6

unknown_20:6$

unknown_21:66

unknown_22:6$

unknown_23:Z$

unknown_24:$$

unknown_25:$$

unknown_26:$$

unknown_27:<

unknown_28:$

unknown_29:

unknown_30:$

unknown_31:

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_191042
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput
?
?	
,__inference_functional_1_layer_call_fn_19796	
input!
unknown:
	unknown_0:#
	unknown_1:
	unknown_2:#
	unknown_3:
	unknown_4:#
	unknown_5:
	unknown_6:#
	unknown_7:$
	unknown_8:$#
	unknown_9:$$

unknown_10:$$

unknown_11:$6

unknown_12:6$

unknown_13:66

unknown_14:6$

unknown_15:6Q

unknown_16:Q$

unknown_17:QQ

unknown_18:Q%

unknown_19:?6

unknown_20:6$

unknown_21:66

unknown_22:6$

unknown_23:Z$

unknown_24:$$

unknown_25:$$

unknown_26:$$

unknown_27:<

unknown_28:$

unknown_29:

unknown_30:$

unknown_31:

unknown_32:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32*.
Tin'
%2#*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:???????????*D
_read_only_resource_inputs&
$"	
 !"*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_196522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*1
_output_shapes
:???????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*t
_input_shapesc
a:???????????: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
1
_output_shapes
:???????????

_user_specified_nameinput"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
A
input8
serving_default_input:0???????????R
CentroidConfmapsHead:
StatefulPartitionedCall:0???????????tensorflow/serving/predict:??	
??
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer_with_weights-2
layer-6
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer-10
layer_with_weights-4
layer-11
layer-12
layer_with_weights-5
layer-13
layer-14
layer-15
layer_with_weights-6
layer-16
layer-17
layer_with_weights-7
layer-18
layer-19
layer-20
layer_with_weights-8
layer-21
layer-22
layer_with_weights-9
layer-23
layer-24
layer-25
layer-26
layer_with_weights-10
layer-27
layer-28
layer_with_weights-11
layer-29
layer-30
 layer-31
!layer-32
"layer_with_weights-12
"layer-33
#layer-34
$layer_with_weights-13
$layer-35
%layer-36
&layer-37
'layer-38
(layer_with_weights-14
(layer-39
)layer-40
*layer_with_weights-15
*layer-41
+layer-42
,layer_with_weights-16
,layer-43
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"??
_tf_keras_network??{"name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "stack0_enc0_conv0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc0_conv0", "inbound_nodes": [[["input", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc0_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc0_act0_relu", "inbound_nodes": [[["stack0_enc0_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc0_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc0_conv1", "inbound_nodes": [[["stack0_enc0_act0_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc0_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc0_act1_relu", "inbound_nodes": [[["stack0_enc0_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc1_pool", "inbound_nodes": [[["stack0_enc0_act1_relu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc1_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc1_conv0", "inbound_nodes": [[["stack0_enc1_pool", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc1_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc1_act0_relu", "inbound_nodes": [[["stack0_enc1_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc1_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc1_conv1", "inbound_nodes": [[["stack0_enc1_act0_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc1_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc1_act1_relu", "inbound_nodes": [[["stack0_enc1_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc2_pool", "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc2_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc2_conv0", "inbound_nodes": [[["stack0_enc2_pool", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc2_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc2_act0_relu", "inbound_nodes": [[["stack0_enc2_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc2_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc2_conv1", "inbound_nodes": [[["stack0_enc2_act0_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc2_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc2_act1_relu", "inbound_nodes": [[["stack0_enc2_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc3_pool", "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc3_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc3_conv0", "inbound_nodes": [[["stack0_enc3_pool", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc3_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc3_act0_relu", "inbound_nodes": [[["stack0_enc3_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc3_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc3_conv1", "inbound_nodes": [[["stack0_enc3_act0_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc3_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc3_act1_relu", "inbound_nodes": [[["stack0_enc3_conv1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc4_last_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc4_last_pool", "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc5_middle_expand_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc5_middle_expand_conv0", "inbound_nodes": [[["stack0_enc4_last_pool", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc5_middle_expand_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc5_middle_expand_act0_relu", "inbound_nodes": [[["stack0_enc5_middle_expand_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_enc6_middle_contract_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc6_middle_contract_conv0", "inbound_nodes": [[["stack0_enc5_middle_expand_act0_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_enc6_middle_contract_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc6_middle_contract_act0_relu", "inbound_nodes": [[["stack0_enc6_middle_contract_conv0", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec0_s16_to_s8_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec0_s16_to_s8_interp_bilinear", "inbound_nodes": [[["stack0_enc6_middle_contract_act0_relu", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "stack0_dec0_s16_to_s8_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec0_s16_to_s8_skip_concat", "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}], ["stack0_dec0_s16_to_s8_interp_bilinear", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec0_s16_to_s8_refine_conv0", "inbound_nodes": [[["stack0_dec0_s16_to_s8_skip_concat", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec0_s16_to_s8_refine_conv1", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0_act_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec1_s8_to_s4_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec1_s8_to_s4_interp_bilinear", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1_act_relu", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "stack0_dec1_s8_to_s4_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec1_s8_to_s4_skip_concat", "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}], ["stack0_dec1_s8_to_s4_interp_bilinear", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec1_s8_to_s4_refine_conv0", "inbound_nodes": [[["stack0_dec1_s8_to_s4_skip_concat", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec1_s8_to_s4_refine_conv1", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0_act_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1", 0, 0, {}]]]}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec2_s4_to_s2_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec2_s4_to_s2_interp_bilinear", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1_act_relu", 0, 0, {}]]]}, {"class_name": "Concatenate", "config": {"name": "stack0_dec2_s4_to_s2_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec2_s4_to_s2_skip_concat", "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}], ["stack0_dec2_s4_to_s2_interp_bilinear", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec2_s4_to_s2_refine_conv0", "inbound_nodes": [[["stack0_dec2_s4_to_s2_skip_concat", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec2_s4_to_s2_refine_conv1", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0_act_relu", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "CentroidConfmapsHead", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CentroidConfmapsHead", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1_act_relu", 0, 0, {}]]]}], "input_layers": [["input", 0, 0]], "output_layers": [["CentroidConfmapsHead", 0, 0]]}, "shared_object_id": 78, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 512, 512, 1]}, "float32", "input"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}, "name": "input", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "stack0_enc0_conv0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc0_conv0", "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Activation", "config": {"name": "stack0_enc0_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc0_act0_relu", "inbound_nodes": [[["stack0_enc0_conv0", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "stack0_enc0_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc0_conv1", "inbound_nodes": [[["stack0_enc0_act0_relu", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Activation", "config": {"name": "stack0_enc0_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc0_act1_relu", "inbound_nodes": [[["stack0_enc0_conv1", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc1_pool", "inbound_nodes": [[["stack0_enc0_act1_relu", 0, 0, {}]]], "shared_object_id": 9}, {"class_name": "Conv2D", "config": {"name": "stack0_enc1_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc1_conv0", "inbound_nodes": [[["stack0_enc1_pool", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Activation", "config": {"name": "stack0_enc1_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc1_act0_relu", "inbound_nodes": [[["stack0_enc1_conv0", 0, 0, {}]]], "shared_object_id": 13}, {"class_name": "Conv2D", "config": {"name": "stack0_enc1_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc1_conv1", "inbound_nodes": [[["stack0_enc1_act0_relu", 0, 0, {}]]], "shared_object_id": 16}, {"class_name": "Activation", "config": {"name": "stack0_enc1_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc1_act1_relu", "inbound_nodes": [[["stack0_enc1_conv1", 0, 0, {}]]], "shared_object_id": 17}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc2_pool", "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Conv2D", "config": {"name": "stack0_enc2_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc2_conv0", "inbound_nodes": [[["stack0_enc2_pool", 0, 0, {}]]], "shared_object_id": 21}, {"class_name": "Activation", "config": {"name": "stack0_enc2_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc2_act0_relu", "inbound_nodes": [[["stack0_enc2_conv0", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Conv2D", "config": {"name": "stack0_enc2_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc2_conv1", "inbound_nodes": [[["stack0_enc2_act0_relu", 0, 0, {}]]], "shared_object_id": 25}, {"class_name": "Activation", "config": {"name": "stack0_enc2_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc2_act1_relu", "inbound_nodes": [[["stack0_enc2_conv1", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc3_pool", "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Conv2D", "config": {"name": "stack0_enc3_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc3_conv0", "inbound_nodes": [[["stack0_enc3_pool", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Activation", "config": {"name": "stack0_enc3_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc3_act0_relu", "inbound_nodes": [[["stack0_enc3_conv0", 0, 0, {}]]], "shared_object_id": 31}, {"class_name": "Conv2D", "config": {"name": "stack0_enc3_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc3_conv1", "inbound_nodes": [[["stack0_enc3_act0_relu", 0, 0, {}]]], "shared_object_id": 34}, {"class_name": "Activation", "config": {"name": "stack0_enc3_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc3_act1_relu", "inbound_nodes": [[["stack0_enc3_conv1", 0, 0, {}]]], "shared_object_id": 35}, {"class_name": "MaxPooling2D", "config": {"name": "stack0_enc4_last_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "stack0_enc4_last_pool", "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}]]], "shared_object_id": 36}, {"class_name": "Conv2D", "config": {"name": "stack0_enc5_middle_expand_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc5_middle_expand_conv0", "inbound_nodes": [[["stack0_enc4_last_pool", 0, 0, {}]]], "shared_object_id": 39}, {"class_name": "Activation", "config": {"name": "stack0_enc5_middle_expand_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc5_middle_expand_act0_relu", "inbound_nodes": [[["stack0_enc5_middle_expand_conv0", 0, 0, {}]]], "shared_object_id": 40}, {"class_name": "Conv2D", "config": {"name": "stack0_enc6_middle_contract_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_enc6_middle_contract_conv0", "inbound_nodes": [[["stack0_enc5_middle_expand_act0_relu", 0, 0, {}]]], "shared_object_id": 43}, {"class_name": "Activation", "config": {"name": "stack0_enc6_middle_contract_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_enc6_middle_contract_act0_relu", "inbound_nodes": [[["stack0_enc6_middle_contract_conv0", 0, 0, {}]]], "shared_object_id": 44}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec0_s16_to_s8_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec0_s16_to_s8_interp_bilinear", "inbound_nodes": [[["stack0_enc6_middle_contract_act0_relu", 0, 0, {}]]], "shared_object_id": 45}, {"class_name": "Concatenate", "config": {"name": "stack0_dec0_s16_to_s8_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec0_s16_to_s8_skip_concat", "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}], ["stack0_dec0_s16_to_s8_interp_bilinear", 0, 0, {}]]], "shared_object_id": 46}, {"class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec0_s16_to_s8_refine_conv0", "inbound_nodes": [[["stack0_dec0_s16_to_s8_skip_concat", 0, 0, {}]]], "shared_object_id": 49}, {"class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0", 0, 0, {}]]], "shared_object_id": 50}, {"class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec0_s16_to_s8_refine_conv1", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 53}, {"class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1", 0, 0, {}]]], "shared_object_id": 54}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec1_s8_to_s4_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec1_s8_to_s4_interp_bilinear", "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 55}, {"class_name": "Concatenate", "config": {"name": "stack0_dec1_s8_to_s4_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec1_s8_to_s4_skip_concat", "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}], ["stack0_dec1_s8_to_s4_interp_bilinear", 0, 0, {}]]], "shared_object_id": 56}, {"class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec1_s8_to_s4_refine_conv0", "inbound_nodes": [[["stack0_dec1_s8_to_s4_skip_concat", 0, 0, {}]]], "shared_object_id": 59}, {"class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0", 0, 0, {}]]], "shared_object_id": 60}, {"class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec1_s8_to_s4_refine_conv1", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 63}, {"class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1", 0, 0, {}]]], "shared_object_id": 64}, {"class_name": "UpSampling2D", "config": {"name": "stack0_dec2_s4_to_s2_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "name": "stack0_dec2_s4_to_s2_interp_bilinear", "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 65}, {"class_name": "Concatenate", "config": {"name": "stack0_dec2_s4_to_s2_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "name": "stack0_dec2_s4_to_s2_skip_concat", "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}], ["stack0_dec2_s4_to_s2_interp_bilinear", 0, 0, {}]]], "shared_object_id": 66}, {"class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec2_s4_to_s2_refine_conv0", "inbound_nodes": [[["stack0_dec2_s4_to_s2_skip_concat", 0, 0, {}]]], "shared_object_id": 69}, {"class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0", 0, 0, {}]]], "shared_object_id": 70}, {"class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 71}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 72}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "stack0_dec2_s4_to_s2_refine_conv1", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 73}, {"class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1", 0, 0, {}]]], "shared_object_id": 74}, {"class_name": "Conv2D", "config": {"name": "CentroidConfmapsHead", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 75}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 76}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CentroidConfmapsHead", "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 77}], "input_layers": [["input", 0, 0]], "output_layers": [["CentroidConfmapsHead", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 512, 512, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input"}}
?

2kernel
3bias
4	variables
5trainable_variables
6regularization_losses
7	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc0_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc0_conv0", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 80}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 1]}}
?
8	variables
9trainable_variables
:regularization_losses
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc0_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc0_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc0_conv0", 0, 0, {}]]], "shared_object_id": 4}
?

<kernel
=bias
>	variables
?trainable_variables
@regularization_losses
A	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc0_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc0_conv1", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc0_act0_relu", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 81}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 512, 512, 16]}}
?
B	variables
Ctrainable_variables
Dregularization_losses
E	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc0_act1_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc0_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc0_conv1", 0, 0, {}]]], "shared_object_id": 8}
?
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc1_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "stack0_enc1_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["stack0_enc0_act1_relu", 0, 0, {}]]], "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 82}}
?

Jkernel
Kbias
L	variables
Mtrainable_variables
Nregularization_losses
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc1_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc1_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 10}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 11}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc1_pool", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}, "shared_object_id": 83}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 16]}}
?
P	variables
Qtrainable_variables
Rregularization_losses
S	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc1_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc1_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc1_conv0", 0, 0, {}]]], "shared_object_id": 13}
?

Tkernel
Ubias
V	variables
Wtrainable_variables
Xregularization_losses
Y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc1_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc1_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 14}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 15}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc1_act0_relu", 0, 0, {}]]], "shared_object_id": 16, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}, "shared_object_id": 84}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 24]}}
?
Z	variables
[trainable_variables
\regularization_losses
]	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc1_act1_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc1_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc1_conv1", 0, 0, {}]]], "shared_object_id": 17}
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc2_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "stack0_enc2_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 85}}
?

bkernel
cbias
d	variables
etrainable_variables
fregularization_losses
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc2_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc2_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc2_pool", 0, 0, {}]]], "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}, "shared_object_id": 86}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 24]}}
?
h	variables
itrainable_variables
jregularization_losses
k	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc2_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc2_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc2_conv0", 0, 0, {}]]], "shared_object_id": 22}
?

lkernel
mbias
n	variables
otrainable_variables
pregularization_losses
q	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc2_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc2_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 23}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 24}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc2_act0_relu", 0, 0, {}]]], "shared_object_id": 25, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 36}}, "shared_object_id": 87}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 36]}}
?
r	variables
strainable_variables
tregularization_losses
u	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc2_act1_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc2_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc2_conv1", 0, 0, {}]]], "shared_object_id": 26}
?
v	variables
wtrainable_variables
xregularization_losses
y	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc3_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "stack0_enc3_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}]]], "shared_object_id": 27, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 88}}
?

zkernel
{bias
|	variables
}trainable_variables
~regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc3_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc3_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc3_pool", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 36}}, "shared_object_id": 89}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 36]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc3_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc3_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc3_conv0", 0, 0, {}]]], "shared_object_id": 31}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "stack0_enc3_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc3_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 32}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 33}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc3_act0_relu", 0, 0, {}]]], "shared_object_id": 34, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 54}}, "shared_object_id": 90}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 54]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc3_act1_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc3_act1_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc3_conv1", 0, 0, {}]]], "shared_object_id": 35}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc4_last_pool", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "stack0_enc4_last_pool", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}]]], "shared_object_id": 36, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 91}}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "stack0_enc5_middle_expand_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc5_middle_expand_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 37}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 38}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc4_last_pool", 0, 0, {}]]], "shared_object_id": 39, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 54}}, "shared_object_id": 92}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 54]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc5_middle_expand_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc5_middle_expand_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc5_middle_expand_conv0", 0, 0, {}]]], "shared_object_id": 40}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_enc6_middle_contract_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_enc6_middle_contract_conv0", "trainable": true, "dtype": "float32", "filters": 81, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 41}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 42}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_enc5_middle_expand_act0_relu", 0, 0, {}]]], "shared_object_id": 43, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 81}}, "shared_object_id": 93}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 81]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_enc6_middle_contract_act0_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_enc6_middle_contract_act0_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_enc6_middle_contract_conv0", 0, 0, {}]]], "shared_object_id": 44}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec0_s16_to_s8_interp_bilinear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "stack0_dec0_s16_to_s8_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "inbound_nodes": [[["stack0_enc6_middle_contract_act0_relu", 0, 0, {}]]], "shared_object_id": 45, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 94}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec0_s16_to_s8_skip_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "stack0_dec0_s16_to_s8_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["stack0_enc3_act1_relu", 0, 0, {}], ["stack0_dec0_s16_to_s8_interp_bilinear", 0, 0, {}]]], "shared_object_id": 46, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 64, 64, 54]}, {"class_name": "TensorShape", "items": [null, 64, 64, 81]}]}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec0_s16_to_s8_refine_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 47}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 48}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec0_s16_to_s8_skip_concat", 0, 0, {}]]], "shared_object_id": 49, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 135}}, "shared_object_id": 95}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 135]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0", 0, 0, {}]]], "shared_object_id": 50}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec0_s16_to_s8_refine_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1", "trainable": true, "dtype": "float32", "filters": 54, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 51}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 52}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 53, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 54}}, "shared_object_id": 96}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 64, 54]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec0_s16_to_s8_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1", 0, 0, {}]]], "shared_object_id": 54}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec1_s8_to_s4_interp_bilinear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "stack0_dec1_s8_to_s4_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "inbound_nodes": [[["stack0_dec0_s16_to_s8_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 55, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 97}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec1_s8_to_s4_skip_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "stack0_dec1_s8_to_s4_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["stack0_enc2_act1_relu", 0, 0, {}], ["stack0_dec1_s8_to_s4_interp_bilinear", 0, 0, {}]]], "shared_object_id": 56, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 128, 128, 36]}, {"class_name": "TensorShape", "items": [null, 128, 128, 54]}]}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec1_s8_to_s4_refine_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 57}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 58}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec1_s8_to_s4_skip_concat", 0, 0, {}]]], "shared_object_id": 59, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 90}}, "shared_object_id": 98}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 90]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0", 0, 0, {}]]], "shared_object_id": 60}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec1_s8_to_s4_refine_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1", "trainable": true, "dtype": "float32", "filters": 36, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 61}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 62}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 63, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 36}}, "shared_object_id": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128, 128, 36]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec1_s8_to_s4_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1", 0, 0, {}]]], "shared_object_id": 64}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec2_s4_to_s2_interp_bilinear", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "stack0_dec2_s4_to_s2_interp_bilinear", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "bilinear"}, "inbound_nodes": [[["stack0_dec1_s8_to_s4_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 65, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 100}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec2_s4_to_s2_skip_concat", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "stack0_dec2_s4_to_s2_skip_concat", "trainable": true, "dtype": "float32", "axis": -1}, "inbound_nodes": [[["stack0_enc1_act1_relu", 0, 0, {}], ["stack0_dec2_s4_to_s2_interp_bilinear", 0, 0, {}]]], "shared_object_id": 66, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 256, 256, 24]}, {"class_name": "TensorShape", "items": [null, 256, 256, 36]}]}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec2_s4_to_s2_refine_conv0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 67}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 68}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec2_s4_to_s2_skip_concat", 0, 0, {}]]], "shared_object_id": 69, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 60}}, "shared_object_id": 101}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 60]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv0_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0", 0, 0, {}]]], "shared_object_id": 70}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?
{"name": "stack0_dec2_s4_to_s2_refine_conv1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1", "trainable": true, "dtype": "float32", "filters": 24, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 71}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 72}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv0_act_relu", 0, 0, {}]]], "shared_object_id": 73, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}, "shared_object_id": 102}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 24]}}
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "stack0_dec2_s4_to_s2_refine_conv1_act_relu", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1", 0, 0, {}]]], "shared_object_id": 74}
?
?kernel
	?bias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "CentroidConfmapsHead", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "CentroidConfmapsHead", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [1, 1]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 75}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 76}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["stack0_dec2_s4_to_s2_refine_conv1_act_relu", 0, 0, {}]]], "shared_object_id": 77, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 24}}, "shared_object_id": 103}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256, 256, 24]}}
?
20
31
<2
=3
J4
K5
T6
U7
b8
c9
l10
m11
z12
{13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
?
20
31
<2
=3
J4
K5
T6
U7
b8
c9
l10
m11
z12
{13
?14
?15
?16
?17
?18
?19
?20
?21
?22
?23
?24
?25
?26
?27
?28
?29
?30
?31
?32
?33"
trackable_list_wrapper
 "
trackable_list_wrapper
?
-	variables
?layer_metrics
?metrics
.trainable_variables
 ?layer_regularization_losses
?non_trainable_variables
?layers
/regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
2:02stack0_enc0_conv0/kernel
$:"2stack0_enc0_conv0/bias
.
20
31"
trackable_list_wrapper
.
20
31"
trackable_list_wrapper
 "
trackable_list_wrapper
?
4	variables
?layer_metrics
5trainable_variables
6regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
8	variables
?layer_metrics
9trainable_variables
:regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:02stack0_enc0_conv1/kernel
$:"2stack0_enc0_conv1/bias
.
<0
=1"
trackable_list_wrapper
.
<0
=1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
>	variables
?layer_metrics
?trainable_variables
@regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
B	variables
?layer_metrics
Ctrainable_variables
Dregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
F	variables
?layer_metrics
Gtrainable_variables
Hregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:02stack0_enc1_conv0/kernel
$:"2stack0_enc1_conv0/bias
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
L	variables
?layer_metrics
Mtrainable_variables
Nregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
P	variables
?layer_metrics
Qtrainable_variables
Rregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:02stack0_enc1_conv1/kernel
$:"2stack0_enc1_conv1/bias
.
T0
U1"
trackable_list_wrapper
.
T0
U1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
V	variables
?layer_metrics
Wtrainable_variables
Xregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Z	variables
?layer_metrics
[trainable_variables
\regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^	variables
?layer_metrics
_trainable_variables
`regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0$2stack0_enc2_conv0/kernel
$:"$2stack0_enc2_conv0/bias
.
b0
c1"
trackable_list_wrapper
.
b0
c1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
d	variables
?layer_metrics
etrainable_variables
fregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
h	variables
?layer_metrics
itrainable_variables
jregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0$$2stack0_enc2_conv1/kernel
$:"$2stack0_enc2_conv1/bias
.
l0
m1"
trackable_list_wrapper
.
l0
m1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
n	variables
?layer_metrics
otrainable_variables
pregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
r	variables
?layer_metrics
strainable_variables
tregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
v	variables
?layer_metrics
wtrainable_variables
xregularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0$62stack0_enc3_conv0/kernel
$:"62stack0_enc3_conv0/bias
.
z0
{1"
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
|	variables
?layer_metrics
}trainable_variables
~regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
2:0662stack0_enc3_conv1/kernel
$:"62stack0_enc3_conv1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
@:>6Q2&stack0_enc5_middle_expand_conv0/kernel
2:0Q2$stack0_enc5_middle_expand_conv0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
B:@QQ2(stack0_enc6_middle_contract_conv0/kernel
4:2Q2&stack0_enc6_middle_contract_conv0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
D:B?62)stack0_dec0_s16_to_s8_refine_conv0/kernel
5:362'stack0_dec0_s16_to_s8_refine_conv0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C:A662)stack0_dec0_s16_to_s8_refine_conv1/kernel
5:362'stack0_dec0_s16_to_s8_refine_conv1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
B:@Z$2(stack0_dec1_s8_to_s4_refine_conv0/kernel
4:2$2&stack0_dec1_s8_to_s4_refine_conv0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
B:@$$2(stack0_dec1_s8_to_s4_refine_conv1/kernel
4:2$2&stack0_dec1_s8_to_s4_refine_conv1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
B:@<2(stack0_dec2_s4_to_s2_refine_conv0/kernel
4:22&stack0_dec2_s4_to_s2_refine_conv0/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
B:@2(stack0_dec2_s4_to_s2_refine_conv1/kernel
4:22&stack0_dec2_s4_to_s2_refine_conv1/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:32CentroidConfmapsHead/kernel
':%2CentroidConfmapsHead/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?	variables
?layer_metrics
?trainable_variables
?regularization_losses
 ?layer_regularization_losses
?non_trainable_variables
?layers
?metrics
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
&37
'38
(39
)40
*41
+42
,43"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
,__inference_functional_1_layer_call_fn_19175
,__inference_functional_1_layer_call_fn_20174
,__inference_functional_1_layer_call_fn_20247
,__inference_functional_1_layer_call_fn_19796?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_functional_1_layer_call_and_return_conditional_losses_20391
G__inference_functional_1_layer_call_and_return_conditional_losses_20535
G__inference_functional_1_layer_call_and_return_conditional_losses_19911
G__inference_functional_1_layer_call_and_return_conditional_losses_20026?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
 __inference__wrapped_model_18573?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input???????????
?2?
1__inference_stack0_enc0_conv0_layer_call_fn_20544?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_20554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc0_act0_relu_layer_call_fn_20559?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_20564?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_stack0_enc0_conv1_layer_call_fn_20573?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_20583?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc0_act1_relu_layer_call_fn_20588?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_20593?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_stack0_enc1_pool_layer_call_fn_18585?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_18579?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_stack0_enc1_conv0_layer_call_fn_20602?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_20612?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc1_act0_relu_layer_call_fn_20617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_20622?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_stack0_enc1_conv1_layer_call_fn_20631?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_20641?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc1_act1_relu_layer_call_fn_20646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_20651?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_stack0_enc2_pool_layer_call_fn_18597?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_18591?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_stack0_enc2_conv0_layer_call_fn_20660?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_20670?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc2_act0_relu_layer_call_fn_20675?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_20680?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_stack0_enc2_conv1_layer_call_fn_20689?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_20699?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc2_act1_relu_layer_call_fn_20704?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_20709?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_stack0_enc3_pool_layer_call_fn_18609?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_18603?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
1__inference_stack0_enc3_conv0_layer_call_fn_20718?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_20728?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc3_act0_relu_layer_call_fn_20733?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_20738?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
1__inference_stack0_enc3_conv1_layer_call_fn_20747?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_20757?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc3_act1_relu_layer_call_fn_20762?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_20767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_stack0_enc4_last_pool_layer_call_fn_18621?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_18615?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
?__inference_stack0_enc5_middle_expand_conv0_layer_call_fn_20776?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_20786?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_stack0_enc5_middle_expand_act0_relu_layer_call_fn_20791?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_20796?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_stack0_enc6_middle_contract_conv0_layer_call_fn_20805?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_20815?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_stack0_enc6_middle_contract_act0_relu_layer_call_fn_20820?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_20825?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_fn_18640?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_18634?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
A__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_fn_20831?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_20838?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_fn_20847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_20857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_fn_20862?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_fn_20876?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_20886?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
K__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_fn_20891?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20896?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_fn_18659?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_18653?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
@__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_fn_20902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_20909?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_fn_20918?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_20928?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_fn_20933?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20938?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_fn_20947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_20957?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_fn_20962?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20967?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_fn_18678?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_18672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
@__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_fn_20973?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_20980?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_fn_20989?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_20999?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_fn_21004?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_21009?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_fn_21018?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_21028?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
J__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_fn_21033?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_21038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
4__inference_CentroidConfmapsHead_layer_call_fn_21047?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_21057?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_20101input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
O__inference_CentroidConfmapsHead_layer_call_and_return_conditional_losses_21057r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
4__inference_CentroidConfmapsHead_layer_call_fn_21047e??9?6
/?,
*?'
inputs???????????
? ""?????????????
 __inference__wrapped_model_18573?623<=JKTUbclmz{????????????????????8?5
.?+
)?&
input???????????
? "U?R
P
CentroidConfmapsHead8?5
CentroidConfmapsHead????????????
G__inference_functional_1_layer_call_and_return_conditional_losses_19911?623<=JKTUbclmz{????????????????????@?=
6?3
)?&
input???????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_20026?623<=JKTUbclmz{????????????????????@?=
6?3
)?&
input???????????
p

 
? "/?,
%?"
0???????????
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_20391?623<=JKTUbclmz{????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? "/?,
%?"
0???????????
? ?
G__inference_functional_1_layer_call_and_return_conditional_losses_20535?623<=JKTUbclmz{????????????????????A?>
7?4
*?'
inputs???????????
p

 
? "/?,
%?"
0???????????
? ?
,__inference_functional_1_layer_call_fn_19175?623<=JKTUbclmz{????????????????????@?=
6?3
)?&
input???????????
p 

 
? ""?????????????
,__inference_functional_1_layer_call_fn_19796?623<=JKTUbclmz{????????????????????@?=
6?3
)?&
input???????????
p

 
? ""?????????????
,__inference_functional_1_layer_call_fn_20174?623<=JKTUbclmz{????????????????????A?>
7?4
*?'
inputs???????????
p 

 
? ""?????????????
,__inference_functional_1_layer_call_fn_20247?623<=JKTUbclmz{????????????????????A?>
7?4
*?'
inputs???????????
p

 
? ""?????????????
#__inference_signature_wrapper_20101?623<=JKTUbclmz{????????????????????A?>
? 
7?4
2
input)?&
input???????????"U?R
P
CentroidConfmapsHead8?5
CentroidConfmapsHead????????????
`__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_and_return_conditional_losses_18634?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
E__inference_stack0_dec0_s16_to_s8_interp_bilinear_layer_call_fn_18640?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
f__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20867h7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
K__inference_stack0_dec0_s16_to_s8_refine_conv0_act_relu_layer_call_fn_20862[7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
]__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_and_return_conditional_losses_20857o??8?5
.?+
)?&
inputs?????????@@?
? "-?*
#? 
0?????????@@6
? ?
B__inference_stack0_dec0_s16_to_s8_refine_conv0_layer_call_fn_20847b??8?5
.?+
)?&
inputs?????????@@?
? " ??????????@@6?
f__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20896h7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
K__inference_stack0_dec0_s16_to_s8_refine_conv1_act_relu_layer_call_fn_20891[7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
]__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_and_return_conditional_losses_20886n??7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
B__inference_stack0_dec0_s16_to_s8_refine_conv1_layer_call_fn_20876a??7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
\__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_and_return_conditional_losses_20838?|?y
r?o
m?j
*?'
inputs/0?????????@@6
<?9
inputs/1+???????????????????????????Q
? ".?+
$?!
0?????????@@?
? ?
A__inference_stack0_dec0_s16_to_s8_skip_concat_layer_call_fn_20831?|?y
r?o
m?j
*?'
inputs/0?????????@@6
<?9
inputs/1+???????????????????????????Q
? "!??????????@@??
___inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_and_return_conditional_losses_18653?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_stack0_dec1_s8_to_s4_interp_bilinear_layer_call_fn_18659?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
e__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_and_return_conditional_losses_20938l9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
J__inference_stack0_dec1_s8_to_s4_refine_conv0_act_relu_layer_call_fn_20933_9?6
/?,
*?'
inputs???????????$
? ""????????????$?
\__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_and_return_conditional_losses_20928r??9?6
/?,
*?'
inputs???????????Z
? "/?,
%?"
0???????????$
? ?
A__inference_stack0_dec1_s8_to_s4_refine_conv0_layer_call_fn_20918e??9?6
/?,
*?'
inputs???????????Z
? ""????????????$?
e__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_and_return_conditional_losses_20967l9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
J__inference_stack0_dec1_s8_to_s4_refine_conv1_act_relu_layer_call_fn_20962_9?6
/?,
*?'
inputs???????????$
? ""????????????$?
\__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_and_return_conditional_losses_20957r??9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
A__inference_stack0_dec1_s8_to_s4_refine_conv1_layer_call_fn_20947e??9?6
/?,
*?'
inputs???????????$
? ""????????????$?
[__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_and_return_conditional_losses_20909?~?{
t?q
o?l
,?)
inputs/0???????????$
<?9
inputs/1+???????????????????????????6
? "/?,
%?"
0???????????Z
? ?
@__inference_stack0_dec1_s8_to_s4_skip_concat_layer_call_fn_20902?~?{
t?q
o?l
,?)
inputs/0???????????$
<?9
inputs/1+???????????????????????????6
? ""????????????Z?
___inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_and_return_conditional_losses_18672?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
D__inference_stack0_dec2_s4_to_s2_interp_bilinear_layer_call_fn_18678?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
e__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_and_return_conditional_losses_21009l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
J__inference_stack0_dec2_s4_to_s2_refine_conv0_act_relu_layer_call_fn_21004_9?6
/?,
*?'
inputs???????????
? ""?????????????
\__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_and_return_conditional_losses_20999r??9?6
/?,
*?'
inputs???????????<
? "/?,
%?"
0???????????
? ?
A__inference_stack0_dec2_s4_to_s2_refine_conv0_layer_call_fn_20989e??9?6
/?,
*?'
inputs???????????<
? ""?????????????
e__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_and_return_conditional_losses_21038l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
J__inference_stack0_dec2_s4_to_s2_refine_conv1_act_relu_layer_call_fn_21033_9?6
/?,
*?'
inputs???????????
? ""?????????????
\__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_and_return_conditional_losses_21028r??9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
A__inference_stack0_dec2_s4_to_s2_refine_conv1_layer_call_fn_21018e??9?6
/?,
*?'
inputs???????????
? ""?????????????
[__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_and_return_conditional_losses_20980?~?{
t?q
o?l
,?)
inputs/0???????????
<?9
inputs/1+???????????????????????????$
? "/?,
%?"
0???????????<
? ?
@__inference_stack0_dec2_s4_to_s2_skip_concat_layer_call_fn_20973?~?{
t?q
o?l
,?)
inputs/0???????????
<?9
inputs/1+???????????????????????????$
? ""????????????<?
P__inference_stack0_enc0_act0_relu_layer_call_and_return_conditional_losses_20564l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_stack0_enc0_act0_relu_layer_call_fn_20559_9?6
/?,
*?'
inputs???????????
? ""?????????????
P__inference_stack0_enc0_act1_relu_layer_call_and_return_conditional_losses_20593l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_stack0_enc0_act1_relu_layer_call_fn_20588_9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_stack0_enc0_conv0_layer_call_and_return_conditional_losses_20554p239?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_stack0_enc0_conv0_layer_call_fn_20544c239?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_stack0_enc0_conv1_layer_call_and_return_conditional_losses_20583p<=9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_stack0_enc0_conv1_layer_call_fn_20573c<=9?6
/?,
*?'
inputs???????????
? ""?????????????
P__inference_stack0_enc1_act0_relu_layer_call_and_return_conditional_losses_20622l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_stack0_enc1_act0_relu_layer_call_fn_20617_9?6
/?,
*?'
inputs???????????
? ""?????????????
P__inference_stack0_enc1_act1_relu_layer_call_and_return_conditional_losses_20651l9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
5__inference_stack0_enc1_act1_relu_layer_call_fn_20646_9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_stack0_enc1_conv0_layer_call_and_return_conditional_losses_20612pJK9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_stack0_enc1_conv0_layer_call_fn_20602cJK9?6
/?,
*?'
inputs???????????
? ""?????????????
L__inference_stack0_enc1_conv1_layer_call_and_return_conditional_losses_20641pTU9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????
? ?
1__inference_stack0_enc1_conv1_layer_call_fn_20631cTU9?6
/?,
*?'
inputs???????????
? ""?????????????
K__inference_stack0_enc1_pool_layer_call_and_return_conditional_losses_18579?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_stack0_enc1_pool_layer_call_fn_18585?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
P__inference_stack0_enc2_act0_relu_layer_call_and_return_conditional_losses_20680l9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
5__inference_stack0_enc2_act0_relu_layer_call_fn_20675_9?6
/?,
*?'
inputs???????????$
? ""????????????$?
P__inference_stack0_enc2_act1_relu_layer_call_and_return_conditional_losses_20709l9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
5__inference_stack0_enc2_act1_relu_layer_call_fn_20704_9?6
/?,
*?'
inputs???????????$
? ""????????????$?
L__inference_stack0_enc2_conv0_layer_call_and_return_conditional_losses_20670pbc9?6
/?,
*?'
inputs???????????
? "/?,
%?"
0???????????$
? ?
1__inference_stack0_enc2_conv0_layer_call_fn_20660cbc9?6
/?,
*?'
inputs???????????
? ""????????????$?
L__inference_stack0_enc2_conv1_layer_call_and_return_conditional_losses_20699plm9?6
/?,
*?'
inputs???????????$
? "/?,
%?"
0???????????$
? ?
1__inference_stack0_enc2_conv1_layer_call_fn_20689clm9?6
/?,
*?'
inputs???????????$
? ""????????????$?
K__inference_stack0_enc2_pool_layer_call_and_return_conditional_losses_18591?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_stack0_enc2_pool_layer_call_fn_18597?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
P__inference_stack0_enc3_act0_relu_layer_call_and_return_conditional_losses_20738h7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
5__inference_stack0_enc3_act0_relu_layer_call_fn_20733[7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
P__inference_stack0_enc3_act1_relu_layer_call_and_return_conditional_losses_20767h7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
5__inference_stack0_enc3_act1_relu_layer_call_fn_20762[7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
L__inference_stack0_enc3_conv0_layer_call_and_return_conditional_losses_20728lz{7?4
-?*
(?%
inputs?????????@@$
? "-?*
#? 
0?????????@@6
? ?
1__inference_stack0_enc3_conv0_layer_call_fn_20718_z{7?4
-?*
(?%
inputs?????????@@$
? " ??????????@@6?
L__inference_stack0_enc3_conv1_layer_call_and_return_conditional_losses_20757n??7?4
-?*
(?%
inputs?????????@@6
? "-?*
#? 
0?????????@@6
? ?
1__inference_stack0_enc3_conv1_layer_call_fn_20747a??7?4
-?*
(?%
inputs?????????@@6
? " ??????????@@6?
K__inference_stack0_enc3_pool_layer_call_and_return_conditional_losses_18603?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_stack0_enc3_pool_layer_call_fn_18609?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
P__inference_stack0_enc4_last_pool_layer_call_and_return_conditional_losses_18615?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
5__inference_stack0_enc4_last_pool_layer_call_fn_18621?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
^__inference_stack0_enc5_middle_expand_act0_relu_layer_call_and_return_conditional_losses_20796h7?4
-?*
(?%
inputs?????????  Q
? "-?*
#? 
0?????????  Q
? ?
C__inference_stack0_enc5_middle_expand_act0_relu_layer_call_fn_20791[7?4
-?*
(?%
inputs?????????  Q
? " ??????????  Q?
Z__inference_stack0_enc5_middle_expand_conv0_layer_call_and_return_conditional_losses_20786n??7?4
-?*
(?%
inputs?????????  6
? "-?*
#? 
0?????????  Q
? ?
?__inference_stack0_enc5_middle_expand_conv0_layer_call_fn_20776a??7?4
-?*
(?%
inputs?????????  6
? " ??????????  Q?
`__inference_stack0_enc6_middle_contract_act0_relu_layer_call_and_return_conditional_losses_20825h7?4
-?*
(?%
inputs?????????  Q
? "-?*
#? 
0?????????  Q
? ?
E__inference_stack0_enc6_middle_contract_act0_relu_layer_call_fn_20820[7?4
-?*
(?%
inputs?????????  Q
? " ??????????  Q?
\__inference_stack0_enc6_middle_contract_conv0_layer_call_and_return_conditional_losses_20815n??7?4
-?*
(?%
inputs?????????  Q
? "-?*
#? 
0?????????  Q
? ?
A__inference_stack0_enc6_middle_contract_conv0_layer_call_fn_20805a??7?4
-?*
(?%
inputs?????????  Q
? " ??????????  Q