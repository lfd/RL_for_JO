SELECT * FROM info_type AS it, company_name AS cn, cast_info AS ci, movie_companies AS mc, aka_name AS an, movie_keyword AS mk, movie_info AS mi, role_type AS rt WHERE ci.note IN ('(voice)', '(voice: Japanese version)', '(voice) (uncredited)', '(voice: English version)') AND cn.country_code = '[us]' AND cn.name = 'DreamWorks Animation' AND it.info = 'release dates' AND mi.info IS NOT NULL AND (mi.info LIKE 'Japan:%201%' OR mi.info LIKE 'USA:%201%') AND rt.role = 'actress' AND mc.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mk.movie_id AND mk.movie_id = mc.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mi.movie_id AND mi.movie_id = mk.movie_id AND mk.movie_id = mi.movie_id AND ci.movie_id = mk.movie_id AND mk.movie_id = ci.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id AND it.id = mi.info_type_id AND mi.info_type_id = it.id AND rt.id = ci.role_id AND ci.role_id = rt.id AND ci.person_id = an.person_id AND an.person_id = ci.person_id;