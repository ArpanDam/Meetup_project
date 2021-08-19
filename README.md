# Meetup_project
Meetup project implementing paper On New Group Popularity Prediction in Event-Based Social Networks : https://ieeexplore.ieee.org/abstract/document/8713911

How to execute the code : To implement the paper : First extract group level features and member level features using the below python files, then find the reduced rank fetaures of the members using matrix factorisation  then find the  neighbours of the member using neighbours.py, then implement the DNN 
Group level features : g2.py - For generating pickle files g2 and g3 of table 1 of the paper
                       g12_13_14.py - For generating pickle files g12, g13 and g14 of table 1 of the paper
Member level features : m4.py - For generating pickle files m4 of table 1 of the paper 
                        m1.py - For generating pickle files m1 of table 1 of the paper 
                        m2.py - For generating pickle files m2 of table 1 of the paper 
                        m7.py - For generating pickle files m7 of table 1 of the paper 
                        m8.py - For generating pickle files m8 of table 1 of the paper 
                        m11.py - For generating pickle files m11 of table 1 of the paper
Reduced rank : matrix_factorisation.py will find the reduced demention of the member level features since here I have implemented 6                        features so reduction of dimention is done from 6 to 4                        
                        
Finding neighbours : neighbours.py To find neighbours of the member of each group

Graphical neural network : DNN : Implement DNN that is fig 6 of the paper will output the predicted accuracy of test labels

Note :  The pickle files which are used as input to generate the group level features, member level features are neighbour.py are not provided here
