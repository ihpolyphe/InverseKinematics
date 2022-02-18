#!/usr/bin/env python
# coding: utf-8
# vs060の順運動学並びに逆運動学を解くソフトを作成する。
from math import cos, sin, pi, sqrt, atan2, acos
import numpy as np

class DHparams:
    #DHパラメータ
    a = [0.0, 0.0, 0.0, 0.305, -0.010, 0.0, 0.0, -0.081]
    alfa = [0, 0., -pi / 2., 0., pi / 2., -pi / 2., pi / 2, 0.]
    d = [0.0, 0.345, 0.0, 0.0, 0.300, 0.0, 0.070, 0.042]
    THETA = [0, 0., -pi / 2, pi / 2, 0., 0., 0., 0.]

class Kinematics():
    def __init__(self):
        self.tools = CalcTools()
        self.dh = DHparams()
        #0判定とするためのしきい値
        self.acos_threshold = 0.001

    def inverse_kinematics(self, target_pose, angle_rad):
        """
        Args:
            target_pos (list) : 目的の位置
            angles_rad (list) : 目的の姿勢

        Returns:
            joint_angle[:num_sols] (list) : 関節角度(複数の場合も)

        """
        #手先の姿勢と位置から６軸の同時変換行列を算出する。
        T06 = self.tools.calcT06(target_pose, angle_rad)
        #逆運動学の解の数
        sum_solutions = 0
        #各関節角度（６関節に対して最大８つの答え）
        joint_angle = np.zeros((8,6))
                
        # q1
        q1 = [0,0]
        X1 = T06[0,3] - T06[0,2] * self.dh.d[6]
        Y1 = T06[1,3] - T06[1,2] * self.dh.d[6]
        arc_tan1 = atan2(Y1, X1) #極座標の場合はatan2を使う
        q1[0] = self.tools.round_rotation(arc_tan1, self.acos_threshold)
        q1[1] = self.tools.round_rotation(arc_tan1 + pi, self.acos_threshold)

        #なぜか逆順処理
        for i in reversed(range(len(q1))):
            s1 =sin(q1[i])
            c1 =cos(q1[i])

            #q3
            q3 = [0,0]
            A3 = X1 * c1 + Y1 *s1
            B3 = T06[2,3] - T06[2,2] * self.dh.d[6] - self.dh.d[1]
            X3 = 2 * self.dh.a[3] * self.dh.d[4]
            Y3 = 2 * self.dh.a[3] * self.dh.a[4]
            Z3 = A3**2 + B3**2 - (self.dh.d[4])**2 - (self.dh.a[3])**2 - (self.dh.a[4])**2
            c3 = Z3 / sqrt(X3 ** 2 + Y3 ** 2)
            
            # when abs(acos(c3)) == pi / 2
            if abs(abs(c3) - 1) < self.acos_threshold:
                c3 = self.tools.fix_for_acos(c3)
            #アークコサインの条件として-1<= c3 <= 1である必要がある。
            elif abs(c3) > 1.:
                print("No solution q3 is got")
                continue    #今回のq3はスキップし、q1のもう一つの解を実行する。
            ac3 = acos(c3)
            atan3 = atan2(Y3, X3)
            q3[0] = self.tools.round_rotation( (ac3 - atan3), self.acos_threshold)
            q3[1] = self.tools.round_rotation((-ac3 - atan3), self.acos_threshold)

            #q2
            for j in reversed(range(len(q3))):
                q2 = [0,0]
                X2 = T06[2,3] -self.dh.d[1] - T06[2,2] * self.dh.d[6]
                Y2 = A3
                Z2 = -self.dh.a[4] * sin(q3[j]) + self.dh.d[4] * cos(q3[j]) + self.dh.a[3]
                c2 = Z2 / sqrt(X2 ** 2 + Y2 ** 2)
                
                # when abs(acos(c2)) == pi / 2
                if abs(abs(c2) - 1) < self.acos_threshold:
                    c2 = self.tools.fix_for_acos(c2)
                #アークコサインの条件として-1<= c2 <= 1である必要がある。
                elif abs(c2) > 1.:
                    print("No solution q2 is got")
                     # 解の無かった要素0を削除する。
                    q3.pop(j)
                    q2.pop(j)
                    continue    #今回のq2はスキップし、q3のもう一つの解を実行する。
                # q3と逆符号の関係
                q2[0] = self.tools.round_rotation(-acos(c2) + atan2(Y2, X2), self.acos_threshold)
                q2[1] = self.tools.round_rotation( acos(c2) + atan2(Y2, X2), self.acos_threshold)

            # q5: q3と同じで二つの解が得られるため
            for j in reversed(range(len(q3))):
                q5 = [0,0]
                c23 = cos(q2[j] + q3[j])
                s23 = sin(q2[j] + q3[j])
                c5 = T06[2,2] * c23 + ( T06[1,2] * s1 + T06[0,2] * c1 ) * s23

                # when abs(acos(c2)) == pi / 2
                if abs(abs(c5) - 1) < self.acos_threshold:
                    c5 = self.tools.fix_for_acos(c5)
                #アークコサインの条件として-1<= c5 <= 1である必要がある。
                elif abs(c5) > 1.0:
                    print("No solution q5 is got")
                    continue    #今回のq2はスキップし、q3のもう一つの解を実行する。 
                q5[0] = self.tools.round_rotation( acos(c5) ,self.acos_threshold)
                q5[1] = self.tools.round_rotation( -acos(c5) ,self.acos_threshold)  
                
                for k in range(len(q5)):
                    s5 = sin(q5[k])
                    if abs(abs(s5) - 1) < self.acos_threshold:
                        print("Gimbal Lock is occuered")
                        q4 = 0.0
                        sgn_c5 = self.tools.fix_for_acos(cos(q5[k]))
                        X6 = T06[1,1] * c1 - T06[0,1] * s1
                        Y6 = c23 * (T06[0,1] * c1 + T06[1,1] * s1) - T06[2,1] * s23
                        atan6 = sgn_c5 * atan2(-Y6, X6)
                        q6 = self.tools.round_rotation(atan6, self.acos_threshold)
                    else:
                        sgn_s5 = self.tools.fix_for_acos(s5)
                        X4 = sgn_s5 * (-s23 * T06[2,2] + c23 * (T06[0,2] * c1 + T06[1,2] * s1) )
                        Y4 = sgn_s5 * (T06[0,2] * s1 - T06[1,2] * c1)
                        q4 = self.tools.round_rotation( atan2(Y4, X4) , self.acos_threshold)

                        X6 = sgn_s5 * (-T06[2,0] * c23 - ( T06[0,0] * c1 + T06[1,0] * s1) * s23)
                        Y6 = sgn_s5 * (-T06[2,1] * c23 - ( T06[0,1] * c1 + T06[1,1] * s1) * s23 )
                        q6 = self.tools.round_rotation( atan2(-Y6, X6), self.acos_threshold)

                    joint_angle[sum_solutions] = np.array([q1[i], q2[j], q3[j], q4, q5[k], q6])
                    sum_solutions +=1
        
        if sum_solutions == 0:
            print("no solutions ")
            return []
        else:
            print("IK solutions is got")
            print(joint_angle[:sum_solutions] )
            return joint_angle[:sum_solutions]
        

    def forward_kinematics(self, joint_angle):
        """順運動学を解く（各関節角度から手先の位置と姿勢を算出）"""
        # 各関節角にハンドの角度（0度）を追加
        joint_angle_with_hand = np.hstack([np.array(joint_angle), 0.00])
        #0リンクの姿勢を定義
        trans = np.identity(4)
        # 各リンクの同時変換行列を取得してT07を求める
        for i in range(1 , 8): #ハンドまでの同時変換行列を作成
            trans = np.dot(trans, self.tools.transformartion_matrix(joint_angle_with_hand[i -1], i))
        # 手先の位置と姿勢を取得
        pose07 = [trans[0,3], trans[1,3], trans[2,3]]
        posture07 = self.tools.calc_ypr(trans)

        return pose07, posture07



class CalcTools:
    def __init__(self):
        self.dh = DHparams()

    def calc_ypr(self, m):
        """ 同次変換行列や回転行列から姿勢の角度を求める関数

        Args:
            m (np.array) : 同次変換行列 or 回転行列

        Returns:
            ypr (np.array) : 姿勢のヨー･ロール･ピッチ角
        """
        y = [0., 0.]
        p = [0., 0.]
        r = [0., 0.]

        px = sqrt(m[2, 1] ** 2 + m[2, 2] ** 2)
        p[0] = atan2(-m[2, 0], +px)
        p[1] = atan2(-m[2, 0], -px)

        if cos(p[0]) == 0.:
            r[0] = atan2(+m[0, 1], m[1, 1])
            r[1] = atan2(-m[0, 1], m[1, 1])
        else:
            r[0] = atan2(+m[1, 0], +m[0, 0])
            r[1] = atan2(-m[1, 0], -m[0, 0])
            y[0] = atan2(+m[2, 1], +m[2, 2])
            y[1] = atan2(-m[2, 1], -m[2, 2])
        ypr = [[y[0], p[0], r[0]], [y[1], p[1], r[1]]]

        return ypr

    def transformartion_matrix(self, theta, i):
        """ DH parametersから関節(i-1)->iにおける同次変換行列を求める

        Args:
            t (float) : 関節角度(rad)
            i (int) : 関節番号

        Returns:
            T (np.array) : 同次変換行列
        """

        theta_calc = theta + self.dh.THETA[i] 
        T = np.array([[cos(theta_calc), -sin(theta_calc), 0., self.dh.a[i]],
                      [cos(self.dh.alfa[i]) * sin(theta_calc), cos(self.dh.alfa[i]) * cos(theta_calc), -sin(self.dh.alfa[i]), -self.dh.d[i] * sin(self.dh.alfa[i]) ],
                      [sin(self.dh.alfa[i]) * sin(theta_calc), sin(self.dh.alfa[i]) * cos(theta_calc), cos(self.dh.alfa[i]), self.dh.d[i] * cos(self.dh.alfa[i]) ],
                      [0., 0., 0., 1.] ])
        return T

    def calcT06(self, pose, posture):
        #位置姿勢を同時変換行列に変換
        U = self.transfer_yrp_matrix(pose, posture)
        #T67を算出
        T67 = self.transformartion_matrix(0.0, 7)
        #T67の逆行列を算出
        T67_inv = np.linalg.inv(T67)
        #U*T67.inv()でT06を算出
        T06 = np.dot(U, T67_inv)
        return T06

    def transfer_yrp_matrix(self, pos, rot_ypr):
        """ 位置と姿勢から同次変換行列を求める

        Args:
            pos (list) : 位置
            rot_ypr (list) : 姿勢

        Returns:
            m  (np.array) : 同次変換行列

        """
        [y, p, r] = rot_ypr

        rot_00 = cos(p) * cos(r)
        rot_01 = sin(p) * cos(r) * sin(y) - sin(r) * cos(y)
        rot_02 = sin(p) * cos(r) * cos(y) + sin(r) * sin(y)
        rot_10 = cos(p) * sin(r)
        rot_11 = sin(p) * sin(r) * sin(y) + cos(r) * cos(y)
        rot_12 = sin(p) * sin(r) * cos(y) - cos(r) * sin(y)
        rot_20 = -sin(p)
        rot_21 = cos(p) * sin(y)
        rot_22 = cos(p) * cos(y)

        m = np.array([[rot_00, rot_01, rot_02, pos[0]],
                      [rot_10, rot_11, rot_12, pos[1]],
                      [rot_20, rot_21, rot_22, pos[2]],
                      [0, 0, 0, 1]])
        return m 

    def round_rotation(self, x, zero_thresh):
        """ 角度を -pi ~ pi に制限する関数

        Args:
            x (float) : 角度

        Returns:
            (float) : 制限された角度

        """
        if abs(x) < zero_thresh:
            return 0.
        elif x < -pi:
            return self.round_rotation(x + 2.0 * pi, zero_thresh)
        elif x > pi:
            return self.round_rotation(x - 2.0 * pi, zero_thresh)
        else:
            return x
    
    def fix_for_acos(self, a):
        """アークコサインの値が0に近いとき、Θがπ/2を返す"""
        if a > 0:
            return 1
        elif a < 0:
            return -1
        elif a == 0:
            return 0
        pass

class Test:
    def __init__(self):
        self.kinematic = Kinematics()
        self.ik()
        # self.fk()

    def fk(self):
        angle_rad =  np.array([0., -1.50525996, 2.68342346, 0., 0.39263282, 0.]).tolist()
        pose07, posture07 = self.kinematic.forward_kinematics(angle_rad)
        print("pose x:{} y:{} z:{}" .format(pose07[0],pose07[1],pose07[2]))
        print("posture yow:{} pitch:{} roll:{}".format(posture07[0][0],posture07[0][1],posture07[0][2]))

    def ik(self):
        trans = [0.08099999823637774, 1.1450447449659166e-17, 0.570000001998]
        posture = [9.01151919422e-09, 1.57079632, 9.01151925545e-09]
        print("joint angles")
        joint_angles = self.kinematic.inverse_kinematics(trans, posture)
        trans_3 = [round(trans[i], 3) for i in range(len(trans))]
        posture_3 = [round(posture[i], 3) for i in range(len(posture))]

        for i, joint_angle in enumerate(joint_angles):
            pose07, posture07 = self.kinematic.forward_kinematics(joint_angle)
            pose07_3 = [round(pose07[j], 3) for j in range(len(pose07))]
            for k in range(len(posture07)):
                posture07_3 = [round(posture07[k][j], 3) for j in range(len(posture07[k]))]
                if pose07_3 == trans_3 and posture07_3 == posture_3:
                    print("The solution using IK is matched at hand position and posture")
                    print("check result: {}".format(i))
                    print("pose x:{} y:{} z:{}" .format(pose07[0],pose07[1],pose07[2]))
                    print("posture yow:{} pitch:{} roll:{}".format(posture07[0][0],posture07[0][1],posture07[0][2]))

def main():
    test = Test()

if __name__ == '__main__':
    main()
    #入力