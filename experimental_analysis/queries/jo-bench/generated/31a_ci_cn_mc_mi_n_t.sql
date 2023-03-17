SELECT * FROM movie_companies AS mc, company_name AS cn, movie_info AS mi, cast_info AS ci, title AS t, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND cn.name LIKE 'Lionsgate%' AND mi.info IN ('Horror', 'Thriller') AND n.gender = 'm' AND t.id = mi.movie_id AND mi.movie_id = t.id AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mi.movie_id AND mi.movie_id = ci.movie_id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND mi.movie_id = mc.movie_id AND mc.movie_id = mi.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cn.id = mc.company_id AND mc.company_id = cn.id;