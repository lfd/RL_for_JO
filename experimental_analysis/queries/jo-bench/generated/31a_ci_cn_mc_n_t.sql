SELECT * FROM movie_companies AS mc, title AS t, company_name AS cn, cast_info AS ci, name AS n WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND cn.name LIKE 'Lionsgate%' AND n.gender = 'm' AND t.id = ci.movie_id AND ci.movie_id = t.id AND t.id = mc.movie_id AND mc.movie_id = t.id AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND n.id = ci.person_id AND ci.person_id = n.id AND cn.id = mc.company_id AND mc.company_id = cn.id;