SELECT * FROM company_name AS cn, movie_companies AS mc, cast_info AS ci WHERE ci.note IN ('(writer)', '(head writer)', '(written by)', '(story)', '(story editor)') AND cn.name LIKE 'Lionsgate%' AND mc.note LIKE '%(Blu-ray)%' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;