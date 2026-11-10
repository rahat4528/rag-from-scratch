"""
Master runner to build all Phase 1 notebooks (00-07).
"""
import course.utils.build_p1_part1 as part1
import course.utils.build_p1_part2 as part2

if __name__ == "__main__":
    print("Building Phase 1 Part 1 (00-03)...")
    part1.build_nb00()
    part1.build_nb01()
    part1.build_nb02()
    part1.build_nb03()
    print("Building Phase 1 Part 2 (04-07)...")
    part2.build_nb04()
    part2.build_nb05()
    part2.build_nb06()
    part2.build_nb07()
    print("All Phase 1 notebooks (00-07) generated successfully!")
