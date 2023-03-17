SELECT * FROM company_name AS cn, movie_companies AS mc, cast_info AS ci WHERE ci.note LIKE '%(voice)%' AND ci.note LIKE '%(uncredited)%' AND cn.country_code = '[ru]' AND ci.movie_id = mc.movie_id AND mc.movie_id = ci.movie_id AND cn.id = mc.company_id AND mc.company_id = cn.id;