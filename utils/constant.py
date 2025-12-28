from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
import numpy as np

kinova_j2n6s300_chain = Chain(
    name="kinova_j2n6s300",
    links=[
        OriginLink(),  # Base link
        URDFLink(
            name="j2n6s300_joint_1",
            origin_translation=np.array([0, 0, 0.15675]),
            origin_orientation=np.array([0, np.pi, 0]),  
            rotation=np.array([0, 0, 1]),
            bounds=(0, 2*np.pi),  
        ),
        URDFLink(
            name="j2n6s300_joint_2",
            origin_translation=np.array([0, 0.0016, -0.11875]),
            origin_orientation=np.array([-np.pi/2, 0, np.pi]),  
            rotation=np.array([0, 0, 1]),
            bounds=(0.8203047484373349, 5.462880558742252), 
        ),
        URDFLink(
            name="j2n6s300_joint_3",
            origin_translation=np.array([0, -0.41, 0]),
            origin_orientation=np.array([0, np.pi, 0]),  
            rotation=np.array([0, 0, 1]),
            bounds=(0.33161255787892263, 5.951572749300664),  
        ),
        URDFLink(
            name="j2n6s300_joint_4",
            origin_translation=np.array([0, 0.2073, -0.0114]),
            origin_orientation=np.array([-np.pi/2, 0, np.pi]),  
            rotation=np.array([0, 0, 1]),
            bounds=(0, 2*np.pi),  
            
        ),
        URDFLink(
            name="j2n6s300_joint_5",
            origin_translation=np.array([0, -0.03703, -0.06414]),
            origin_orientation=np.array([1.0471975511965976, 0, np.pi]),  
            rotation=np.array([0, 0, 1]),
            bounds=(0, 2*np.pi),  
            
        ),
        URDFLink(
            name="j2n6s300_joint_6",
            origin_translation=np.array([0, -0.03703, -0.06414]),
            origin_orientation=np.array([1.0471975511965976, 0, np.pi]),
            rotation=np.array([0, 0, 1]),
            bounds=(0, 2*np.pi), 
        ),
        URDFLink(
            name="j2n6s300_end_effector",
            origin_translation=np.array([0, 0, 0]),
            origin_orientation=np.array([np.pi, 0, np.pi/2]), 
            rotation=np.array([0, 0, 0]), 
        ),
    ],
)


kinova_initial_position = np.array([
    0.0,        
    4.79607201, 
    4.02273655, 
    0.48501062, 
    2.6347928, 
    2.32984471, 
    2.34214023, 
    0.0         
])

kinova_reset_joint_angles = np.array([
    5.72668171,
    2.90796757,
    0.93086213,
    4.39843321,
    1.20793521,
    2.08407688
])